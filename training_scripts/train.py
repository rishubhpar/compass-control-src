import sys 
import argparse
import hashlib
import itertools
import math
import os
import shutil 
import os.path as osp 
import inspect
from pathlib import Path
from typing import Optional

import copy 

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np 
from io import BytesIO

from utils import * 

import matplotlib.pyplot as plt 
import textwrap 
from distutils.util import strtobool 

import pickle 

from custom_attention_processor import patch_custom_attention 

from infer import TOKEN2ID, UNIQUE_TOKENS 

DEBUG = False  
PRINT_STUFF = False  
BS = 4         
SAVE_STEPS_GAP = 5000  

from datasets import CompassControlDataset  


from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs 
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from lora_diffusion import (
    extract_lora_ups_down,
    inject_trainable_lora,
    safetensors_available,
    save_lora_weight,
    save_safeloras,
)

import cv2 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from pathlib import Path

import random
import re

from compass_encoder import CompassEncoder  
# from viewpoint_mlp import viewpoint_MLP_light_21_multi as viewpoint_MLP

import glob

logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="the run name",
    )
    parser.add_argument(
        "--controlnet_prompts_file",
        type=str,
        default=None,
        required=True,
        help="path to the txt file containing prompts for controlnet augmentation",
    )
    parser.add_argument(
        "--root_data_dir",
        type=str,
        default=None,
        required=True,
        help="root data directory",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir_1subject",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_data_dir_2subjects",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images of single subject",
    )
    parser.add_argument(
        "--controlnet_data_dir_1subject",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of from controlnet.",
    )
    parser.add_argument(
        "--controlnet_data_dir_2subjects",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of from controlnet.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--normalize_merged_embedding", 
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="whether to normalize the merged embedding", 
    )
    parser.add_argument(
        "--use_ref_images", 
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="whether to use the reference (black bg) images", 
    )
    parser.add_argument(
        "--use_controlnet_images", 
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="whether to use the reference (black bg) images", 
    )
    parser.add_argument(
        "--replace_attn_maps", 
        type=str, 
        choices=["special2class", "special2class_detached", "class2special", "class2special_detached", "class2special_soft"], 
        help="whether to replace the special token attention maps by the class token attention maps", 
    ) 
    parser.add_argument(
        "--with_prior_preservation",
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        required=True, 
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        help="the directory where the intermediate visualizations and inferences are stored",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["pt", "safe", "both"],
        default="both",
        help="The output format of the model predicitions and checkpoints.",
    )
    parser.add_argument(
        "--seed", type=int, default=1908, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--color_jitter",
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="Whether to apply color jitter to images",
    )
    parser.add_argument(
        "--train_unet",
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="Whether to train the unet",
    )
    parser.add_argument(
        "--train_text_encoder",
        type=lambda x : bool(strtobool(x)),  
        required=True, 
        help="Whether to train the text encoder",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA approximation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_text",
        type=float,
        default=None, 
        help="Initial learning rate for text encoder (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_mlp",
        type=float,
        default=None, 
        help="Initial learning rate for mlp (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_emb",
        type=float,
        default=None, 
        help="Initial learning rate for embedding (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_merger",
        type=float,
        default=None, 
        help="Initial learning rate for merger (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--stage1_steps",
        type=int,
        required=True, 
        help="Number of steps for stage 1 training", 
    )
    parser.add_argument(
        "--stage2_steps",
        type=int,
        required=True, 
        help="Number of steps for stage 2 training", 
    )
    parser.add_argument(
        "--merged_emb_dim",
        type=int,
        required=True, 
        help="the output dimension of the merger",  
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--resume_training_state",
        type=str,
        default=None,
        help=("File path for unet lora to resume training."),
    )
    # parser.add_argument(
    #     "--resume_unet",
    #     type=str,
    #     default=None,
    #     help=("File path for unet lora to resume training."),
    # )
    # parser.add_argument(
    #     "--resume_text_encoder",
    #     type=str,
    #     default=None,
    #     help=("File path for text encoder lora to resume training."),
    # )
    parser.add_argument(
        "--resize",
        type=bool,
        default=True,
        required=False,
        help="Should images be resized to --resolution before training?",
    )
    parser.add_argument(
        "--use_xformers", action="store_true", help="Whether or not to use xformers"
    )
    
    

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # if args.with_prior_preservation:
    #     if args.class_data_dir is None:
    #         raise ValueError("You must specify a data directory for class images.")
    # else:
    #     if args.class_data_dir is not None:
    #         logger.warning(
    #             "You need not use --class_data_dir without --with_prior_preservation."
    #         )
    #     if args.class_prompt is not None:
    #         logger.warning(
    #             "You need not use --class_prompt without --with_prior_preservation."
    #         )

    if not safetensors_available:
        if args.output_format == "both":
            print(
                "Safetensors is not available - changing output format to just output PyTorch files"
            )
            args.output_format = "pt"
        elif args.output_format == "safe":
            raise ValueError(
                "Safetensors is not available - either install it, or change output_format."
            )

    return args


def main(args): 

    # the single subject images 
    if osp.exists(args.instance_data_dir_1subject): 
        # subjects_ are the folders in the instance directory 
        subjects_combs_1subject = sorted(os.listdir(args.instance_data_dir_1subject))  
        # args.subjects_combs_1subject = [" ".join(subjects_comb.split("__")) for subjects_comb in subjects_combs_1subject]  
        args.subjects_combs_1subject = subjects_combs_1subject 

    # the two subject images 
    if osp.exists(args.instance_data_dir_2subjects): 
        subjects_combs_2subjects = sorted(os.listdir(args.instance_data_dir_2subjects))  
        # args.subjects_combs_2subjects = [" ".join(subjects_comb.split("__")) for subjects_comb in subjects_combs_2subjects]  
        args.subjects_combs_2subjects = subjects_combs_2subjects  

    # defining the output directory to store checkpoints 
    args.output_dir = osp.join(args.output_dir, f"__{args.run_name}") 

    args.subjects = os.listdir(args.instance_data_dir_1subject) 

    # storing the number of reference images per subject 
    args.n_ref_imgs = {} 
    if osp.exists(args.instance_data_dir_1subject): 
        for subject_comb_ in args.subjects_combs_1subject: 
            img_files = os.listdir(osp.join(args.instance_data_dir_1subject, subject_comb_)) 
            img_files = [img_file for img_file in img_files if img_file.find("jpg") != -1] 
            args.n_ref_imgs[subject_comb_] = len(img_files)  

    if osp.exists(args.instance_data_dir_2subjects): 
        for subject_comb_ in args.subjects_combs_2subjects: 
            img_files = os.listdir(osp.join(args.instance_data_dir_2subjects, subject_comb_)) 
            img_files = [img_file for img_file in img_files if img_file.find("jpg") != -1] 
            args.n_ref_imgs[subject_comb_] = len(img_files)  

    # max train steps 
    args.max_train_steps = args.stage1_steps + args.stage2_steps + 1  

    # accelerator 
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    # effective batch size should remain constant 
    assert accelerator.num_processes * args.train_batch_size * args.gradient_accumulation_steps == BS, f"{accelerator.num_processes = }, {args.train_batch_size = }" 


    if args.resume_training_state is not None: 
        assert osp.exists(args.resume_training_state) 
        training_state_ckpt = torch.load(args.resume_training_state) 

    if (
        args.train_text_encoder
        and args.gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # setting a different seed for each process to increase diversity in minibatch 
    set_seed(args.seed + accelerator.process_index) 

    # Handle the repository creation
    # handle the creation of output directory 
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )

    accelerator.wait_for_everyone() 

    if args.resume_training_state is not None: 
        assert osp.exists(args.resume_training_state) 
        training_state_ckpt = torch.load(args.resume_training_state) 

    if accelerator.is_main_process: 
        pkl_path = osp.join(args.output_dir, f"args.pkl") 
        with open(pkl_path, "wb") as f: 
            pickle.dump(args.__dict__, f) 

    SAVE_STEPS = [500, 1000, 5000]  
    for save_step in range(SAVE_STEPS_GAP, args.max_train_steps + 1, SAVE_STEPS_GAP): 
        SAVE_STEPS.append(save_step) 
    SAVE_STEPS = sorted(SAVE_STEPS) 

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=None if args.pretrained_vae_name_or_path else args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    unet.requires_grad_(False)
    if args.train_unet: 
        unet_lora_params, _ = inject_trainable_lora(
            unet, r=args.lora_rank  
        )
        
        # sanity checks 
        n_lora_params_state_dict = 0 
        for name, param in unet.state_dict().items(): 
            if name.find("lora") != -1: 
                n_lora_params_state_dict += 1 

        assert n_lora_params_state_dict == len(unet_lora_params) 

        if args.resume_training_state: 
            # with torch.no_grad(): 
            unet_state_dict = unet.state_dict() 
            lora_state_dict = training_state_ckpt["unet"]["lora"] 
            for name, param in unet_state_dict.items(): 
                if name.find("lora") == -1: 
                    assert name not in lora_state_dict.keys() 
                    continue 
                assert name in lora_state_dict.keys() 
                unet_state_dict[name] = lora_state_dict[name]  
            unet.load_state_dict(unet_state_dict) 

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False) 

    # injecting trainable lora in text encoder 
    if args.train_text_encoder:
        text_encoder_lora_params, _ = inject_trainable_lora(
            text_encoder,
            target_replace_module=["CLIPAttention"],
            r=args.lora_rank,
        )

        # sanity checks 
        n_lora_params_state_dict = 0 
        for name, param in text_encoder.state_dict(): 
            if name.find("lora") != -1: 
                n_lora_params_state_dict += 1 
        assert n_lora_params_state_dict == len(text_encoder_lora_params) 

        if args.resume_training_state: 
            text_encoder_state_dict = text_encoder.state_dict() 
            lora_state_dict = training_state_ckpt["text_encoder"]["lora"] 
            for name, param in text_encoder_state_dict.items():  
                if name.find("lora") == -1: 
                    assert name not in lora_state_dict 
                    continue 
                assert name in lora_state_dict  
                text_encoder_state_dict[name] = lora_state_dict[name]  
            text_encoder.load_state_dict(text_encoder_state_dict) 


    if args.use_xformers:
        set_use_memory_efficient_attention_xformers(unet, True)
        set_use_memory_efficient_attention_xformers(vae, True)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()


    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizers = {}  
    if args.train_unet: 
        optimizer_unet = optimizer_class(
            itertools.chain(*unet_lora_params), 
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        if args.resume_training_state: 
            optimizer_unet.load_state_dict(training_state_ckpt["unet"]["optimizer"]) 
        # optimizers.append(optimizer_unet) 
        optimizers["unet"] = optimizer_unet 

    if args.train_text_encoder: 
        optimizer_text_encoder = optimizer_class(
            itertools.chain(*text_encoder_lora_params),  
            lr=args.learning_rate_text,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
        if args.resume_training_state: 
            optimizer_text_encoder.load_state_dict(training_state_ckpt["text_encoder"]["optimizer"]) 
        # optimizers.append(optimizer_text_encoder) 
        optimizers["text_encoder"] = optimizer_text_encoder 

    merger = CompassEncoder(output_dim=args.merged_emb_dim)  

    if args.resume_training_state: 
        merger.load_state_dict(training_state_ckpt["merger"]["model"], strict=False)  
    optimizer_merger = optimizer_class(
        merger.parameters(),  
        lr=args.learning_rate_merger,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    if args.resume_training_state: 
        optimizer_merger.load_state_dict(training_state_ckpt["merger"]["optimizer"]) 
    optimizers["merger"] = optimizer_merger 


    noise_scheduler = DDPMScheduler.from_config(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # defining the dataset 
    train_dataset_stage1 = CompassControlDataset(
        args=args, 
        tokenizer=tokenizer, 
        ref_imgs_dirs=[args.instance_data_dir_1subject], 
        controlnet_imgs_dirs=[args.controlnet_data_dir_1subject], 
        num_steps=args.stage1_steps * args.train_batch_size, 
        gpu_idx=accelerator.process_index, 
    ) 

    train_dataset_stage2 = CompassControlDataset(
        args=args, 
        tokenizer=tokenizer, 
        ref_imgs_dirs=[args.instance_data_dir_1subject, args.instance_data_dir_2subjects],  
        controlnet_imgs_dirs=[args.controlnet_data_dir_1subject, args.controlnet_data_dir_2subjects],  
        num_steps=args.stage2_steps * args.train_batch_size, 
        gpu_idx=accelerator.process_index, 
    ) 

    def collate_fn(examples):
        is_controlnet = [example["controlnet"] for example in examples] 
        prompt_ids = [example["prompt_ids"] for example in examples] 
        prompts = [example["prompt"] for example in examples] 
        subjects = [example["subjects"] for example in examples] 
        bboxes = [example["bboxes"] for example in examples] 
        xs_2d = [example["2d_xs"] for example in examples] 
        ys_2d = [example["2d_ys"] for example in examples] 
        pixel_values = []
        for example in examples:
            pixel_values.append(example["img"])

        """Adding the scaler of the embedding into the batch"""
        scalers = [example["scalers"] for example in examples] 

        if args.with_prior_preservation:
            prompt_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_img"] for example in examples]
            prompts += [example["class_prompt"] for example in examples] 
            prior_subjects = [example["prior_subject"] for example in examples] 

        pixel_values = torch.stack(pixel_values) 
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float() 

        prompt_ids = torch.stack(prompt_ids, dim=0).to(torch.long) 

        batch = {
            "prompt_ids": prompt_ids, 
            "pixel_values": pixel_values,
            "scalers": scalers,
            "subjects": subjects, 
            "controlnet": is_controlnet, 
            "prompts": prompts, 
            "2d_xs": xs_2d, 
            "2d_ys": ys_2d, 
            "bboxes": bboxes, 
        }
        if args.with_prior_preservation: 
            batch["prior_subjects"] = prior_subjects  

        return batch 

    train_dataloader_stage1 = torch.utils.data.DataLoader(
        train_dataset_stage1,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=accelerator.num_processes * 2,
    )

    train_dataloader_stage2 = torch.utils.data.DataLoader(
        train_dataset_stage2,
        batch_size=args.train_batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=accelerator.num_processes, 
    ) 

    unet, text_encoder, merger = accelerator.prepare(unet, text_encoder, merger)    
    optimizers_ = {} 
    for name, optimizer in optimizers.items(): 
        optimizer = accelerator.prepare(optimizer) 
        # optimizers_.append(optimizer) 
        optimizers_[name] = optimizer 
    optimizers = optimizers_  

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    for name, param in unet.state_dict().items(): 
        if name.find("lora") == -1: 
            param.to(accelerator.device, dtype=weight_dtype) 
    
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    global_step = 0
    
    if args.train_unet: 
        unet.train()
    if args.train_text_encoder: 
        text_encoder.train() 

    merger.train() 

    # steps_per_angle = {} 
    input_embeddings = torch.clone(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight).detach()  

    train_dataloader_stage1_iter = iter(train_dataloader_stage1) 
    train_dataloader_stage2_iter = iter(train_dataloader_stage2) 


    while True: 

        retval = patch_custom_attention(accelerator.unwrap_model(unet))  
        if args.resume_training_state: 
            if global_step < training_state_ckpt["global_step"]:  
                global_step += 1   
                progress_bar.update(1) 
                continue 

        if global_step <= args.stage1_steps:  
            MAX_SUBJECTS_PER_EXAMPLE = 1  
            batch = next(train_dataloader_stage1_iter)  
            progress_bar.set_description(f"Stage 1 training, step {global_step}") 
        else: 
            MAX_SUBJECTS_PER_EXAMPLE = 2   
            batch = next(train_dataloader_stage2_iter)  
            progress_bar.set_description(f"Stage 2 training, step {global_step}") 

        if DEBUG: 
            assert torch.allclose(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight, input_embeddings) 

        B = len(batch["scalers"])   

        if PRINT_STUFF: 
            accelerator.print(f"<=============================== step {global_step}  ======================================>")
            for key, value in batch.items(): 
                if ("ids" in key) or ("values" in key): 
                    print(f"{key}: ", end="") 
                    accelerator.print(f"{value.shape}") 
                else:
                    accelerator.print(f"{key}: {value}") 
            accelerator.print(f"{MAX_SUBJECTS_PER_EXAMPLE = }") 

        # Convert images to latent space
        vae.to(accelerator.device, dtype=weight_dtype)  

        latents = vae.encode(
            batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)  
        ).latent_dist.sample() 
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # if we are in stage 2 of training, only then do we need to compute the pose embedding, otherwise it is zero 
        # if global_step > args.stage1_steps: 
        # we are no longer learning appearance embeddings first! 
        scalers_padded = torch.zeros((len(batch["scalers"]), MAX_SUBJECTS_PER_EXAMPLE))  
        xs_2d_padded = torch.zeros((len(batch["2d_xs"]), MAX_SUBJECTS_PER_EXAMPLE)) 
        ys_2d_padded = torch.zeros((len(batch["2d_ys"]), MAX_SUBJECTS_PER_EXAMPLE)) 

        for batch_idx in range(len(batch["scalers"])): 
            for scaler_idx in range(len(batch["scalers"][batch_idx])): 
                scalers_padded[batch_idx][scaler_idx] = batch["scalers"][batch_idx][scaler_idx] 

        assert len(batch["scalers"]) == len(batch["2d_xs"]) == len(batch["2d_ys"])  
        for batch_idx in range(len(batch["scalers"])): 
            assert len(batch["scalers"][batch_idx]) == len(batch["2d_xs"][batch_idx]) == len(batch["2d_ys"][batch_idx])  
            for scaler_idx in range(len(batch["scalers"][batch_idx])): 
                xs_2d_padded[batch_idx][scaler_idx] = batch["2d_xs"][batch_idx][scaler_idx]  
                ys_2d_padded[batch_idx][scaler_idx] = batch["2d_ys"][batch_idx][scaler_idx] 

        p = torch.Tensor(scalers_padded / (2 * math.pi)) 
        assert torch.all(xs_2d_padded < 1) 
        assert torch.all(ys_2d_padded < 1) 
        if PRINT_STUFF: 
            accelerator.print(f"{p = }") 
            accelerator.print(f"{xs_2d_padded = }") 
            accelerator.print(f"{ys_2d_padded = }") 
        assert torch.all(p <= 1.0) and torch.all(p >= 0.0) 
        assert p.shape == (B, MAX_SUBJECTS_PER_EXAMPLE), f"{p.shape = }" 
        mlp_emb = merger(p) 

        # mlp_emb = torch.cat(mlp_emb, dim=1) 
        assert mlp_emb.shape == (B, MAX_SUBJECTS_PER_EXAMPLE, args.merged_emb_dim) 
        assert args.merged_emb_dim % 1024 == 0, f"{args.merged_emb_dim = }" 

        num_assets_in_batch = 0 
        for batch_idx in range(B): 
            num_assets_in_batch = num_assets_in_batch + len(batch["scalers"][batch_idx]) 

        merged_emb = mlp_emb 
        assert merged_emb.shape[0] == B 

        # replacing the input embedding for sks by the mlp for each batch item, and then getting the output embeddings of the text encoder 
        # must run a for loop here, first changing the input embeddings of the text encoder for each 
        encoder_hidden_states = [] 
        attn_assignments = [] 
        if args.with_prior_preservation: 
            input_ids, input_ids_prior = torch.chunk(batch["prompt_ids"], 2, dim=0) 
        else: 
            input_ids = batch["prompt_ids"] 

        for batch_idx, batch_item in enumerate(input_ids): 
            # replacing the text encoder input embeddings by the original ones and setting them to be COLD -- to enable replacement by a hot embedding  
            accelerator.unwrap_model(text_encoder).get_input_embeddings().weight = torch.nn.Parameter(torch.clone(input_embeddings), requires_grad=False)  

            # performing the replacement on cold embeddings by a hot embedding -- allowed 
            example_merged_emb = merged_emb[batch_idx] 
            for asset_idx, subject in enumerate(batch["subjects"][batch_idx]):   
                for token_idx in range(args.merged_emb_dim // 1024):  
                    # replacement_emb = torch.clone(merged_emb[batch_idx][asset_idx][token_idx * 1024 : (token_idx+1) * 1024])  
                    if args.normalize_merged_embedding: 
                        replacement_mask = torch.ones_like(example_merged_emb, requires_grad=False)      
                        replacement_emb_norm = torch.linalg.norm(example_merged_emb[asset_idx][token_idx * 1024 : (token_idx+1) * 1024]).detach()   
                        org_emb_norm = torch.linalg.norm(accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[subject]]).detach()  
                        replacement_mask[asset_idx][token_idx * 1024 : (token_idx+1) * 1024] = org_emb_norm / replacement_emb_norm  
                        assert example_merged_emb.shape == replacement_mask.shape  
                        assert torch.allclose(torch.linalg.norm((example_merged_emb * replacement_mask)[asset_idx][token_idx * 1024 : (token_idx+1) * 1024]), org_emb_norm, atol=1e-3), f"{torch.linalg.norm((example_merged_emb * replacement_mask)[asset_idx][token_idx * 1024 : (token_idx+1) * 1024]) = }, {org_emb_norm = }" 
                        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[UNIQUE_TOKENS[f"{asset_idx}_{token_idx}"]]] = (example_merged_emb * replacement_mask)[asset_idx][token_idx * 1024 : (token_idx+1) * 1024] 
                    else: 
                        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[TOKEN2ID[UNIQUE_TOKENS[f"{asset_idx}_{token_idx}"]]] = (example_merged_emb)[asset_idx][token_idx * 1024 : (token_idx+1) * 1024] 

            text_embeddings = text_encoder(batch_item.unsqueeze(0).to(accelerator.device))[0].squeeze() 

            attn_assignments_batchitem = {} 
            unique_token_positions = {}  
            for asset_idx in range(len(batch["subjects"][batch_idx])):  
                for token_idx in range(args.merged_emb_dim // 1024): 
                    unique_token = UNIQUE_TOKENS[f"{asset_idx}_{token_idx}"] 
                    assert TOKEN2ID[unique_token] in list(batch_item), f"{unique_token = }" 
                    unique_token_idx = list(batch_item).index(TOKEN2ID[unique_token]) 
                    attn_assignments_batchitem[unique_token_idx] = unique_token_idx + args.merged_emb_dim // 1024 - token_idx 
                    unique_token_positions[f"{asset_idx}_{token_idx}"] = unique_token_idx  

            attn_assignments.append(attn_assignments_batchitem) 
            encoder_hidden_states.append(text_embeddings)  

        encoder_hidden_states = torch.stack(encoder_hidden_states)  

        # replacing the text encoder input embeddings by the original ones, this time setting them to be HOT, this will be useful in case we choose to do textual inversion 
        # here we are not cloning because these won't be stepped upon anyways, and this way we can save some memory also!  
        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight = torch.nn.Parameter(torch.clone(input_embeddings), requires_grad=False)   
        if args.with_prior_preservation: 
            encoder_hidden_states_prior = text_encoder(input_ids_prior.to(accelerator.device))[0] 
            assert encoder_hidden_states_prior.shape == encoder_hidden_states.shape 
            encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_prior], dim=0) 
            assert len(input_ids_prior) == B, f"{len(input_ids_prior) = }, {args.train_batch_size = }" 
            for _ in range(args.train_batch_size):  
                attn_assignments.append({}) 


        encoder_states_dict = {
            "encoder_hidden_states": encoder_hidden_states, 
            "attn_assignments": attn_assignments, 
        } 
        if args.replace_attn_maps is not None: 
            encoder_states_dict[args.replace_attn_maps] = True 

        encoder_states_dict["bbox_from_class_mean"] = True 
        encoder_states_dict["bboxes"] = batch["bboxes"] 

        # if args.replace_attn_maps is not None or args.penalize_special_token_attn or args.bbox_from_class_mean:  
        if DEBUG: 
            os.makedirs(osp.join("vis_attnmaps", f"{str(global_step).zfill(3)}"), exist_ok=True) 
        model_pred = unet(noisy_latents, timesteps, encoder_states_dict).sample 
        # else: 
        #     model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )

        losses = [] 
        if args.with_prior_preservation:
            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            losses.append(loss.detach()) 

            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
            losses.append(prior_loss.detach() * args.prior_loss_weight) 

            # Add the prior loss to the instance loss.
            loss = loss + args.prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            losses.append(loss.detach()) 
            losses.append(torch.tensor(0.0).to(accelerator.device)) 

        if PRINT_STUFF: 
            accelerator.print(f"MSE loss: {losses[0].item()}, the weight is 1.0")
            accelerator.print(f"prior loss: {losses[1].item()}, {args.prior_loss_weight = }") 

        losses = torch.stack(losses).to(accelerator.device) 

        accelerator.backward(loss)
        for name, optimizer in optimizers.items(): 
            optimizer.step() 

        # calculating weight norms 
        progress_bar.update(1) 

        for name, optimizer in optimizers.items(): 
            optimizer.zero_grad() 

        global_step += 1  

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            # if args.save_steps and global_step - last_save >= args.save_steps:
            if len(SAVE_STEPS) > 0 and global_step >= SAVE_STEPS[0]: 
                save_step = SAVE_STEPS[0] 
                SAVE_STEPS.pop(0) 
                if accelerator.is_main_process:
                    accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
                        inspect.signature(
                            accelerator.unwrap_model
                        ).parameters.keys()
                    )
                    extra_args = (
                        {"keep_fp32_wrapper": True}
                        if accepts_keep_fp32_wrapper
                        else {}
                    )
                    pipeline = StableDiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=accelerator.unwrap_model(unet, **extra_args),
                        text_encoder=accelerator.unwrap_model(
                            text_encoder, **extra_args
                        ),
                        revision=args.revision,
                    )

                    training_state = {} 
                    training_state["global_step"] = global_step 
                    training_state["appearance"] = {} 
                    training_state["merger"] = {} 
                    training_state["text_encoder"] = {} 
                    training_state["unet"] = {} 

                    if args.train_unet: 
                        unet_lora_state_dict = {} 
                        for name, param in accelerator.unwrap_model(unet).state_dict().items(): 
                            if name.find(f"lora") == -1: 
                                continue 
                            unet_lora_state_dict[name] = param 

                    if args.train_text_encoder: 
                        text_encoder_lora_state_dict = {} 
                        for name, param in accelerator.unwrap_model(text_encoder).state_dict().items(): 
                            if name.find(f"lora") == -1: 
                                continue 
                            text_encoder_lora_state_dict[name] = param 

                    training_state["merger"]["optimizer"] = optimizers["merger"].state_dict() 
                    training_state["merger"]["model"] = accelerator.unwrap_model(merger).state_dict() 

                    if args.train_unet: 
                        training_state["unet"]["optimizer"] = optimizers["unet"].state_dict() 
                        training_state["unet"]["model"] = args.pretrained_model_name_or_path  
                        # training_state["unet"]["lora"] = list(itertools.chain(*unet_lora_params)) 
                        training_state["unet"]["lora"] = unet_lora_state_dict  

                    if args.train_text_encoder: 
                        training_state["text_encoder"]["optimizer"] = optimizers["text_encoder"].state_dict() 
                        training_state["text_encoder"]["model"] = args.pretrained_model_name_or_path  
                        # training_state["text_encoder"]["lora"] = list(itertools.chain(*text_encoder_lora_params)) 
                        training_state["text_encoder"]["lora"] = text_encoder_lora_state_dict  

                    save_dir = osp.join(args.output_dir, f"training_state_{global_step}.pth")
                    torch.save(training_state, save_dir)   

                    accelerator.print(f"<=========== SAVED CHECKPOINT FOR STEP {global_step} ===============>") 

        loss = loss.detach()
        gathered_loss = torch.mean(accelerator.gather(loss), dim=0)
        # on gathering the list of losses, the shape will be (G, 2) if there are 2 losses 
        # mean along the zeroth dimension would give the actual losses 
        losses = losses.unsqueeze(0) 
        logs = {"loss": gathered_loss.item()} 

        progress_bar.set_postfix(**logs)
        # accelerator.log(logs, step=global_step)

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    controlnet_prompts = []
    prompts_file = open(args.controlnet_prompts_file)
    for line in prompts_file.readlines():
        prompt = str(line).strip() 
        controlnet_prompts.append(prompt)
    args.controlnet_prompts = controlnet_prompts 
    main(args)