#from .transformer_utils import BertAttention, trans_nd, layer_norm
from transformers import AutoConfig
# Import t-net 
import sys
import sys
sys.path.append('/content/GuidedSymbolicGPT/src/ModelSchema')  # Add base src directory
from GPT_schema import BaseGPTConfig, BaseGPT
from transformerBlock import TransformerBlock
sys.path.append('/content/GuidedSymbolicGPT/src/') 
from tokenizer_ops import OPS_Tokenizer, OPS_Tokenizer_Config
from ModelSchema.pointNets.TNet import tNet, tNetConfig
from point_net_schema import BasePointNetConfig
from improved_diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    ModelMeanType,
    ModelVarType,
    LossType,
    get_named_beta_schedule
)



# from transformers import BertEncoder
from transformers.models.bert.modeling_bert import BertEncoder
import torch
from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    timestep_embedding,
    checkpoint,
)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x



class TransSimpleBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        config=None,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        attention_head_size = 64
        assert self.out_channels % attention_head_size == 0
        self.in_layers = nn.Sequential(
            layer_norm(channels),
            SiLU(),
            trans_nd(config, channels, self.out_channels // attention_head_size, attention_head_size),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            layer_norm(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                trans_nd(config, self.out_channels, self.out_channels // attention_head_size, attention_head_size),
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:

            self.skip_connection = trans_nd(config, channels, self.out_channels // attention_head_size,
                                            attention_head_size)
        else:
            self.skip_connection = nn.Sequential(nn.Linear(self.channels, self.out_channels),
                                                 nn.LayerNorm(self.out_channels, eps=config.layer_norm_eps),
                                                 )

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        # print('-'*30)
        # print(self.in_layers)
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        # print(self.in_layers, h.shape, x.shape, )
        # print(emb.shape, self.emb_layers, emb_out.shape)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out.unsqueeze(1)
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=-1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h





class TransModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=1,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        config=None,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if config is None:
            config = AutoConfig.from_pretrained('bert-base-uncased')
            config.position_embedding_type = 'relative_key'
            config.max_position_embeddings = 256

            # print(self.position_embedding_type, config.max_position_embeddings)


        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        attention_head_size = 64
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    trans_nd(config, in_channels, model_channels // attention_head_size, attention_head_size)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    TransformerBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        config=config,
                    )
                ]
                ch = mult * model_channels
                # if ds in attention_resolutions:
                #     layers.append(
                #         AttentionBlock(
                #             ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                #         )
                #     )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            TransformerBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                config=config,
            ),
            # AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            TransformerBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                config=config,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    TransformerBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        config=config,
                    )
                ]
                ch = model_channels * mult
                # if ds in attention_resolutions:
                #     layers.append(
                #         AttentionBlock(
                #             ch,
                #             use_checkpoint=use_checkpoint,
                #             num_heads=num_heads_upsample,
                #         )
                #     )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        attention_head_size_final = 8
        self.out = nn.Sequential(
            layer_norm(ch),
            SiLU(),
            trans_nd(config, model_channels, out_channels // attention_head_size_final, attention_head_size_final),
        # zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        print(self.out, out_channels)

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=-1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)

    def get_feature_vectors(self, x, timesteps, y=None):
        """
        Apply the model and return all of the intermediate tensors.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = th.cat([h, hs.pop()], dim=-1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result

class SymbolicTransformer(nn.Module):
    def __init__(self, tnet_config, transformer_config, tokenizer):
        super(SymbolicTransformer, self).__init__()
        self.tokenizer = tokenizer

        # Initialize TNet for point embedding
        self.point_embedding = tNet(**tnet_config)

        # Transformer layers for sequence modeling
        self.embedding_dim = tnet_config.get("embedding_dim", 128)
        self.transformer = nn.Transformer(
            d_model=self.embedding_dim,
            nhead=transformer_config["nhead"],
            num_encoder_layers=transformer_config["num_encoder_layers"],
            num_decoder_layers=transformer_config["num_decoder_layers"],
            dim_feedforward=transformer_config["dim_feedforward"],
            dropout=transformer_config.get("dropout", 0.1),
            activation=transformer_config.get("activation", "relu")
        )

        # Token classification head
        self.output_vocab_size = tokenizer.vocab_size
        self.output_projection = nn.Linear(self.embedding_dim, self.output_vocab_size)

    def forward(self, point_cloud, target_formula_tokens=None):
        """
        Forward pass of the model.

        Args:
            point_cloud (torch.Tensor): Input point cloud of shape (batch_size, num_points, point_dim).
            target_formula_tokens (torch.Tensor, optional): Target tokens for teacher forcing in training.

        Returns:
            torch.Tensor: Predicted logits for the formula tokens.
        """
        # Normalize and embed the point cloud using TNet
        point_cloud = (point_cloud - point_cloud.mean(dim=1, keepdim=True)) / point_cloud.std(dim=1, keepdim=True)
        point_features = self.point_embedding(point_cloud)  # Shape: (batch_size, num_points, embedding_dim)

        # Prepare transformer inputs
        encoder_input = point_features.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, embedding_dim)

        if target_formula_tokens is not None:
            target_embedded = nn.functional.one_hot(target_formula_tokens, num_classes=self.output_vocab_size).float()
            target_embedded = target_embedded.permute(1, 0, 2)  # Shape: (seq_len, batch_size, embedding_dim)

            # Transformer forward pass
            transformer_output = self.transformer(
                src=encoder_input,
                tgt=target_embedded
            )
        else:
            transformer_output = self.transformer(src=encoder_input)

        # Project transformer output to vocabulary logits
        logits = self.output_projection(transformer_output)  # Shape: (seq_len, batch_size, vocab_size)
        return logits


class TransformerNetModel2(nn.Module):
    def __init__(
        self,
        in_channels,  # This is n_var (without target)
        model_channels,  # This is n_embd
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        config=None,
        config_name='bert-base-uncased',
        training_mode='points',
        vocab_size=None,
        experiment_mode='lm',
        init_pretrained=False,
        logits_mode=1,
    ):
        super().__init__()

        # Initialize t-net with correct input dimensions
        self.tnet_config = tNetConfig(
            n_var=in_channels,  # Original variable count (without target)
            n_embd=model_channels
        )
        self.tnet = tNet(self.tnet_config)

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout

        # Store model parameters
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout_rate = dropout  # Renamed to avoid confusion with dropout layer
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample if num_heads_upsample != -1 else num_heads
        self.vocab_size = vocab_size

        # Initialize formula prediction head
        self.lm_head = nn.Linear(model_channels, vocab_size)

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        # Class embedding if using class conditioning
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # Input projection (from T-Net embedding dimension to transformer hidden size)
        self.input_up_proj = nn.Sequential(
            nn.Linear(model_channels, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        # Initialize transformer
        self.input_transformers = BertEncoder(config)

        # Position embeddings
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Output projection (back to embedding dimension)
        self.output_down_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, out_channels)
        )

    def get_point_embeds(self, points):
        print(f"Input points shape: {points.shape}")
        embeddings = self.tnet(points)
        print(f"Output embeddings shape: {embeddings.shape}")
        return embeddings

    def get_logits(self, hidden_repr):
        """
        Convert hidden representations to formula predictions.
        
        Args:
            hidden_repr: tensor of shape [batch_size, seq_len, n_embd]
        Returns:
            logits: tensor of shape [batch_size, seq_len, vocab_size]
        """
        logits = self.lm_head(hidden_repr)  # Linear layer projecting to vocab size
        print(f"logits shape: {logits.shape}") 
        return logits

    def forward(self, x, timesteps, y=None):
        """
        Forward pass through the model.
        
        Args:
            x: tensor of shape [batch_size, n_embd]
               (output from get_point_embeds that's been noised)
            timesteps: tensor of shape [batch_size] with diffusion timesteps
            y: optional class labels
        Returns:
            output: tensor of shape [batch_size, n_embd]
        """
        # Project embedding to transformer dimension
        emb_x = self.input_up_proj(x)  # [batch_size, hidden_size]
        
        # Add time embeddings
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        # Reshape for transformer if needed (expects sequence dimension)
        if len(emb_x.shape) == 2:
            emb_x = emb_x.unsqueeze(1)  # [batch_size, 1, hidden_size]
            time_emb = time_emb.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Add positional embeddings
        seq_length = emb_x.size(1)
        position_ids = self.position_ids[:, :seq_length]
        emb_inputs = (
            self.position_embeddings(position_ids) + 
            emb_x + 
            time_emb
        )

        # Apply transformer
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        hidden_states = self.input_transformers(emb_inputs).last_hidden_state

        # Project output back to embedding dimension
        output = self.output_down_proj(hidden_states)
        
        # Remove sequence dimension if it was added
        if output.size(1) == 1:
            output = output.squeeze(1)
            
        return output.type(x.dtype)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformers import BertEncoder, AutoConfig

# Find this class in your transformer_model2.py file:
class SymbolicDiffusionTransformer(nn.Module):
    def __init__(
        self,
        tnet_config,
        betas,  # Add this parameter
        hidden_size=256,
        num_layers=6,
        num_heads=8,
        vocab_size=512,
        dropout=0.1,
        tokenizer=None
    ):
        super().__init__()

        self.tokenizer = tokenizer
        
        # Initialize t-net for point cloud encoding
        self.tnet = tNet(tnet_config)
        self.n_embd = tnet_config.n_embd
        
        # Store the provided betas
        alphas = 1.0 - betas
        self.register_buffer('alphas', torch.from_numpy(alphas).float())
        self.register_buffer('alphas_cumprod', torch.from_numpy(np.cumprod(alphas, axis=0)).float())
        
        # BERT configuration
        self.config = AutoConfig.from_pretrained('bert-base-uncased')
        self.config.hidden_size = hidden_size
        self.config.num_hidden_layers = num_layers
        self.config.num_attention_heads = num_heads
        self.config.hidden_dropout_prob = dropout
        self.config.attention_probs_dropout_prob = dropout
        self.config.max_position_embeddings = 512
        self.config.type_vocab_size = 1
        
        # Embedding projections
        self.input_proj = nn.Sequential(
            nn.Linear(self.n_embd, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )
        
        # Time embedding
        time_dim = hidden_size * 4
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # Main transformer
        self.encoder = BertEncoder(self.config)
        
        # Output heads
        self.noise_pred = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.n_embd)
        )
        
        self.lm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_point_embeds(self, points):
        """Get embeddings from point cloud data."""
        # Ensure points are on the correct device
        return self.tnet(points)

    def get_logits(self, point_embeds):
        """Convert embeddings to token logits."""
        batch_size = point_embeds.shape[0]
        hidden_states = self.input_proj(point_embeds)
        hidden_states = hidden_states.unsqueeze(1)  # Add sequence dimension
        hidden_states = hidden_states + self.pos_embed
        
        attention_mask = torch.ones(
            hidden_states.shape[0], 1,
            device=hidden_states.device
        )
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = encoder_outputs.last_hidden_state
        
        # Ensure output shape is [batch_size * seq_len, vocab_size]
        logits = self.lm_head(hidden_states)  # [batch_size, seq_len, vocab_size]
        return logits.view(-1, logits.size(-1))

    def forward(self, points, timesteps, return_logits=False):
        if return_logits:
            point_embeds = self.get_point_embeds(points)
            batch_size = points.shape[0]
            seq_len = 15  # Fixed sequence length
            
            # Project to transformer hidden size
            hidden_states = self.input_proj(point_embeds)
            hidden_states = hidden_states.unsqueeze(1).expand(-1, seq_len, -1)
            hidden_states = hidden_states + self.pos_embed
            
            # Create proper attention mask
            attention_mask = torch.ones(
                batch_size, seq_len,
                device=hidden_states.device
            )
            
            # Convert attention mask to the format expected by BERT
            # [batch_size, 1, 1, seq_length]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=hidden_states.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            
            # Pass through transformer
            encoder_outputs = self.encoder(
                hidden_states,
                attention_mask=extended_attention_mask,
                return_dict=True
            )
            hidden_states = encoder_outputs.last_hidden_state
            
            # Get logits for each position
            logits = self.lm_head(hidden_states)  # [batch_size, seq_len, vocab_size]
            return logits
        
        # For noise prediction
        point_embeds = self.get_point_embeds(points)
        hidden_states = self.input_proj(point_embeds)
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states = hidden_states + self.pos_embed
        
        # Add time embeddings
        if timesteps is not None:
            time_emb = timestep_embedding(timesteps, self.config.hidden_size)
            time_emb = self.time_embed(time_emb)
            hidden_states = hidden_states + time_emb.unsqueeze(1)
        
        # Create proper attention mask for single timestep
        attention_mask = torch.ones(
            hidden_states.shape[0], 1,
            device=hidden_states.device
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=hidden_states.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Pass through transformer
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            return_dict=True
        )
        hidden_states = encoder_outputs.last_hidden_state
        
        # Return noise prediction
        return self.noise_pred(hidden_states)

    def generate(self, points, temperature=1, max_length=15):
        """Generate formulas using autoregressive sampling."""
        self.eval()
        batch_size = points.shape[0]
        device = points.device
        
        # Initialize with first token
        generated = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
        
        with torch.no_grad():
            for i in range(max_length):
                # Get logits for next token
                logits = self(points, torch.zeros(batch_size, device=device), return_logits=True)
                current_logits = logits[:, i, :]  # Get logits for current position
                
                # Apply temperature
                current_logits = current_logits / temperature
                
                # Sample next token
                probs = F.softmax(current_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)[:, 0]
                
                # Add to sequence
                generated[:, i] = next_token
                
                # Early stopping if all sequences have hit 'N' token
                if i > 2 and (next_token == self.tokenizer.token_list.index('N')).all():
                    break
        
        return generated

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# --- Test Case for TransformerNetModel2 ---
# This block tests the functionality of the TransformerNetModel2
# with dummy inputs for point cloud embeddings and diffusion timesteps.

# Dummy Inputs
# batch_size = 16
# num_points = 100
# in_channels = 128
# model_channels = 128
# seq_length = 20

# dummy_points = torch.randn(batch_size, in_channels, num_points)  # Shape: (B, in_channels, num_points)
# dummy_timesteps = torch.randint(0, 1000, (batch_size,))  # Random timesteps for diffusion
# dummy_tokens = torch.randint(0, 512, (batch_size, seq_length))  # Random symbolic tokens

# # Initialize model
# tnet_config = tNetConfig(n_var=in_channels, n_embd=model_channels)
# transformer_config = {
#     "nhead": 8,
#     "num_encoder_layers": 6,
#     "num_decoder_layers": 6,
#     "dim_feedforward": 512,
#     "dropout": 0.1
# }
# config_name = "bert-base-uncased"
# vocab_size = 512

# model = TransformerNetModel2(
#     in_channels=in_channels,
#     model_channels=model_channels,
#     out_channels=vocab_size,
#     num_res_blocks=2,
#     attention_resolutions=[4],
#     dropout=0.1,
#     config_name=config_name,
#     vocab_size=vocab_size,
# )

# # Forward Pass
# output = model(dummy_points, dummy_timesteps)
# print("Output shape:", output.shape)  # Expect: (B, seq_length, vocab_size)
