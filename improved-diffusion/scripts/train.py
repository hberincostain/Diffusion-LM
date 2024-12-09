"""
Train a diffusion model for text generation.
"""

import argparse
import json
import torch
import os
from improved_diffusion import dist_util, logger
from improved_diffusion.text_datasets import load_data_text
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from transformers import AutoTokenizer, set_seed
from improved_diffusion.train_util import TrainLoop
import wandb

def main():
    args = create_argparser().parse_args()
    set_seed(args.seed) 
    
    # Initialize distributed environment
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'the parameter count is {pytorch_total_params}')
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    os.makedirs(args.checkpoint_path, exist_ok=True)
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "diffusion_lm"),
        name=args.checkpoint_path,
    )
    wandb.config.update(args.__dict__, allow_val_change=True)

    logger.log("creating data loader...")
    data = load_data_text(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        data_args=args,
        task_mode='e2e-tgt',
        padding_mode=args.padding_mode,
        load_vocab=None,  # Will be loaded from tokenizer
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_interval=args.eval_interval
    ).run_loop()

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=50,
        save_interval=50000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        seed=101,
        gradient_clipping=-1.0,
        eval_interval=2000,
        checkpoint_path='diffusion_models',
        # Text-specific defaults
        modality='e2e-tgt',
        config_name='bert-base-uncased',
        vocab_size=821,
        image_size=8,
        num_channels=128,
        num_res_blocks=2,
        dropout=0.1,
        model_arch='transformer',
        experiment_mode='lm',
        padding_mode='block',
        # Add e2e training parameter
        e2e_train='../datasets/e2e_data',
        predict_xstart=True,
        training_mode='e2e',
        in_channel=16,
        out_channel=16,
        experiment='random',
        use_kl=False,
        learn_sigma=False,
        timestep_respacing="",
        use_checkpoint=False,
        use_scale_shift_norm=True,
        logits_mode=1
    )
    
    defaults.update(model_and_diffusion_defaults())
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
