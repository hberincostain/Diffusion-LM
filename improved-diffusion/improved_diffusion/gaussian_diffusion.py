import enum
import math
import numpy as np
import torch as th
from torch import nn

# ----------------------- Utility Functions -----------------------

def normal_kl(mean1, logvar1, mean2, logvar2):
    # Dummy function for demonstration, not a true KL calculation
    return mean_flat((mean1 - mean2)**2 + (th.exp(logvar1) - th.exp(logvar2)))

def discretized_text_log_likelihood(x, logits):
    # Dummy function for demonstration, just returns a placeholder value
    return mean_flat(-th.log_softmax(logits, dim=-1).gather(dim=-1, index=x.unsqueeze(-1)).squeeze(-1))

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "sqrt":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1 - np.sqrt(t + 0.0001),
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

class ModelVarType(enum.Enum):
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()

class LossType(enum.Enum):
    POINT_MSE = enum.auto()  # MSE for point cloud embeddings
    POINT_KL = enum.auto()   # KL for point cloud embeddings
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

# ----------------------- GaussianDiffusion Class -----------------------

class GaussianDiffusion:
    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        # Keep existing initialization code
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Process beta schedule
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means no diffusion)
        """
        if noise is None:
            noise = th.randn_like(x_start)
        
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_mean_variance(self, model, x_t, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t)
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, D = x_t.shape
        assert t.shape == (B,)

        # Direct prediction of noise
        model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

        # Convert model output to mean and variance
        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x_t, t=t
            )
        else:  # EPSILON case
            pred_xstart = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x_t, t=t
            )

        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)

        # Get variance
        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(self, model, x_t, t, noise_fn=None, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Sample p(x_{t-1} | x_t)
        """
        out = self.p_mean_variance(
            model,
            x_t,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = noise_fn or th.randn_like
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise(x_t)
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        Get x_0 prediction from epsilon
        """
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def training_losses(self, model, x_start, t, points=None, target_formula=None, model_kwargs=None):
        """
        Compute training losses for a single timestep.
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Add noise to embeddings
        noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        # Get model prediction
        model_output = model(points, t, **model_kwargs)

        if self.model_mean_type == ModelMeanType.START_X:
            target = x_start
        else:
            target = noise

        # MSE loss for diffusion
        mse_loss = mean_flat((target - model_output) ** 2)

        # Optional token prediction loss
        token_loss = 0.0
        if target_formula is not None:
            logits = model(points, t * 0, return_logits=True)  # Use t=0 for clean predictions
            token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_formula.view(-1),
                reduction='mean'
            )

        loss = mse_loss + token_loss

        return {
            "loss": loss,
            "mse": mse_loss,
            "token_loss": token_loss,
        }

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

# ----------------------- Inference Helper -----------------------

def generate_embeddings(diffusion_model, model, initial_noise, num_timesteps):
    x_t = initial_noise
    # Diffusion loop
    for step in reversed(range(num_timesteps)):
        t = th.tensor([step], device=x_t.device, dtype=th.long)
        with th.no_grad():
            output = diffusion_model.p_mean_variance(
                model=model,
                x=x_t,
                t=t,
                clip_denoised=True
            )
            # Sample from the distribution using mean - For simplicity, we ignore variance sampling here.
            # In a real scenario, you'd add noise according to model_variance for a stochastic sampling.
            x_t = output["pred_xstart"]

    return x_t  # These are embeddings

def generate_tokens(diffusion_model, model, initial_noise, num_timesteps):
    # First generate final embeddings from the diffusion process
    final_embeddings = generate_embeddings(diffusion_model, model, initial_noise, num_timesteps)
    # Convert embeddings to logits
    logits = model.get_logits(final_embeddings)
    # Argmax to get tokens
    tokens = th.argmax(logits, dim=-1)
    return tokens

# ----------------------- Example Usage -----------------------
if __name__ == "__main__":
    # Initialize the Gaussian Diffusion model
    betas = get_named_beta_schedule("linear", num_diffusion_timesteps=1000)
    diffusion_model = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,  # predicting epsilon
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.POINT_MSE,
    )

    # Dummy transformer model that:
    # - Predicts epsilon in the same embedding dimension as x_t
    # After diffusion, we use another linear layer to get logits.
    class DummyTransformer(nn.Module):
        def __init__(self, embedding_dim, vocab_size):
            super().__init__()
            self.embedding_dim = embedding_dim
            self.vocab_size = vocab_size

            # Projection to predict eps (same dim as embedding)
            self.eps_projection = nn.Linear(embedding_dim, embedding_dim)

            # Separate projection for converting embeddings to logits after diffusion
            self.to_logits = nn.Linear(embedding_dim, vocab_size)

        def forward(self, x, t, **kwargs):
            # x: (batch, seq_len, embedding_dim)
            # We'll just return a predicted epsilon of the same shape
            return self.eps_projection(x)

        def get_logits(self, x):
            # Convert embeddings to logits
            return self.to_logits(x)

    model = DummyTransformer(embedding_dim=128, vocab_size=512)

    # Dummy inputs: random embeddings as initial noise
    initial_noise = th.randn(16, 100, 128)  # (batch_size, seq_length, embedding_dim)
    tokens = generate_tokens(diffusion_model, model, initial_noise, num_timesteps=1000)
    print("Generated Tokens Shape:", tokens.shape)
