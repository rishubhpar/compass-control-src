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
import pickle

import copy

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np
from io import BytesIO

from utils import *
from compass_encoder import *

import matplotlib.pyplot as plt

sys.path.append(f"..")
from lora_diffusion import patch_pipe
# from metrics import MetricEvaluator from safetensors.torch import load_file

WHICH_LATENTS = "1"
WHICH_MODEL = "compass"
# WHICH_MODEL = "replace_attn_maps"
WHICH_STEP = 20000  
MAX_SUBJECTS_PER_EXAMPLE = 2
NUM_SAMPLES = 16


KEYWORD = f""

NUM_INFERENCE_STEPS = 50

INSTANCE_DIR_1SUBJECT = "../training_data_2subjects_3009/ref_imgs_1subject"
INSTANCE_DIR_2SUBJECTS = "../training_data_2subjects_3009/ref_imgs_2subjects"

from custom_attention_processor import patch_custom_attention, get_attention_maps, show_image_relevance

TOKEN2ID = {
    "sks": 48136,
    "bnha": 49336,
    "pickup truck": 4629, # using the token for "truck" instead
    "bus": 2840,
    "cat": 2368,
    "giraffe": 22826,
    "horse": 4558,
    "lion": 5567,
    "elephant": 10299,
    "jeep": 11286,
    "motorbike": 33341,
    "bicycle": 11652,
    "tractor": 14607,
    "truck": 4629,
    "zebra": 22548,
    "sedan": 24237,
    "suv": 15985,
    "motocross": 34562,
    "boat": 4440,
    "ship": 1158,
    "plane":5363,
    "helicopter": 11956,
    "shoe": 7342,
    "bird": 3329,
    "sparrow": 22888,
    "suitcase": 27792,
    "chair": 4269,
    "dolphin": 16464,
    "fish": 2759,
    "shark": 7980,
    "man": 786,
    "camel": 21914,
    "dog": 1929,
    "pickup": 15382,

    # unque tokens
    "bk": 14083,
    "ak": 1196,
    "ck": 868,
    "dk": 16196,
    "ek": 2092,
    "fk": 12410,
    "gk": 18719,
}

# UNIQUE_TOKENS = ["bnha", "sks", "ak", "bk", "ck", "dk", "ek", "fk", "gk"]
UNIQUE_TOKENS = {
    "0_0": "bnha",
    "0_1": "sks",
    "0_2": "ak",
    "1_0": "bk",
    "1_1": "ck",
    "1_2": "dk",
    "2_0": "ek",
    "2_1": "fk",
    "2_2": "gk",
}


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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pathlib import Path

import random
import re

from compass_encoder import CompassEncoder 
# from viewpoint_mlp import viewpoint_MLP_light_21_multi as viewpoint_MLP

import glob
import wandb


class EncoderStatesDataset(Dataset):
    def __init__(self, encoder_states, save_paths, attn_assignments, track_ids, interesting_token_strs, all_azimuths, all_subjects, all_bboxes_per_frame): # Added all_bboxes_per_frame
        assert len(encoder_states) == len(save_paths) == len(track_ids) == len(attn_assignments) == len(interesting_token_strs) == len(all_azimuths) == len(all_subjects) == len(all_bboxes_per_frame) > 0 # Added assertion for all_bboxes_per_frame
        self.encoder_states = encoder_states
        self.save_paths = save_paths
        self.attn_assignments = attn_assignments
        self.track_ids = track_ids
        self.interesting_token_strs = interesting_token_strs
        self.azimuths = all_azimuths
        self.subjects = all_subjects
        self.bboxes_per_frame = all_bboxes_per_frame # Store bboxes

    def __len__(self):
       return len(self.encoder_states)

    def __getitem__(self, index):
        assert self.save_paths[index] is not None
        assert self.encoder_states[index] is not None
        # print(f"dataset is sending {self.encoder_states[index] = }, {self.save_paths[index] = }")
        # (self.encoder_states[index], [self.save_paths[index]])
        return (self.encoder_states[index], self.save_paths[index], self.attn_assignments[index], self.track_ids[index], self.interesting_token_strs[index], self.azimuths[index], self.subjects[index], self.bboxes_per_frame[index]) # Return bboxes


def collate_fn(examples):
    save_paths = [example[1] for example in examples]
    encoder_states = torch.stack([example[0] for example in examples], 0)
    attn_assignments = [example[2] for example in examples]
    track_ids = [example[3] for example in examples]
    interesting_token_strs = [example[4] for example in examples]
    azimuths = [example[5] for example in examples]
    subjects = [example[6] for example in examples]
    bboxes_per_frame = [example[7] for example in examples] # Collect bboxes
    return {
        "save_paths": save_paths,
        "encoder_states": encoder_states,
        "attn_assignments": attn_assignments,
        "track_ids": track_ids,
        "interesting_token_strs": interesting_token_strs,
        "azimuths": azimuths,
        "subjects": subjects,
        "bboxes_per_frame": bboxes_per_frame, # Add bboxes to the output dictionary
    }


class Infer:
    def __init__(self, merged_emb_dim, accelerator, unet, scheduler, vae, text_encoder, tokenizer, merger, tmp_dir, bs=8):
        self.merged_emb_dim = merged_emb_dim
        self.accelerator = accelerator
        self.unet = unet
        self.bs = bs
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.vae = vae
        self.merger = merger
        self.tmp_dir = tmp_dir

        self.tokenizer = tokenizer

        self.unet = self.accelerator.prepare(self.unet)
        self.text_encoder = self.accelerator.prepare(self.text_encoder)
        self.scheduler = self.accelerator.prepare(self.scheduler)
        self.vae = self.accelerator.prepare(self.vae)
        self.merger = self.accelerator.prepare(self.merger)
        # assert not osp.exists(self.gif_name)


        img_transforms = []
        img_transforms.append(
            transforms.Resize(
                512, interpolation=transforms.InterpolationMode.BILINEAR
            )
        )

        self.image_transforms = transforms.Compose(
            [*img_transforms, transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )


    def generate_images_in_a_batch_and_save_them(self, batch, step_idx):
        # print(f"{self.accelerator.process_index} is doing {batch_idx = }")
        uncond_tokens = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        uncond_encoder_states = self.text_encoder(uncond_tokens.to(self.accelerator.device))[0]
        uncond_assignments = []
        uncond_subjects = []
        uncond_azimuths = []
        uncond_bboxes = [] # Bounding boxes for unconditional generation (typically empty)

        B_actual = batch["encoder_states"].shape[0] # Actual batch size for this iteration
        for _ in range(B_actual): # Use B_actual here
            uncond_assignments.append({})
            uncond_subjects.append([])
            uncond_azimuths.append([])
            uncond_bboxes.append([]) # Append empty list for each unconditional sample's bboxes

        cond_assignments = batch["attn_assignments"]
        cond_subjects = batch["subjects"]
        cond_azimuths = batch["azimuths"]
        # Bboxes from batch are already lists of tensors, potentially on CPU
        cond_bboxes_per_frame_from_batch = batch["bboxes_per_frame"]

        # Move conditional bboxes to device
        cond_bboxes_per_frame_on_device = []
        for frame_bboxes_list in cond_bboxes_per_frame_from_batch:
            tensor_list_on_device = [
                bbox.to(self.accelerator.device, dtype=torch.float32) for bbox in frame_bboxes_list
            ]
            cond_bboxes_per_frame_on_device.append(tensor_list_on_device)

        all_assignments = uncond_assignments + cond_assignments
        all_subjects = uncond_subjects + cond_subjects
        all_azimuths = uncond_azimuths + cond_azimuths
        all_bboxes = uncond_bboxes + cond_bboxes_per_frame_on_device # Combined bboxes
        assert len(all_assignments) == len(all_subjects) == len(all_azimuths) == len(all_bboxes)

        encoder_states = batch["encoder_states"].to(self.accelerator.device)
        save_paths = batch["save_paths"]
        print(f"{self.accelerator.process_index} is doing {save_paths}")
        B = encoder_states.shape[0] # This B is B_actual
        assert encoder_states.shape == (B, 77, 1024)
        if self.seed is not None:
            set_seed(self.seed)
        latents = torch.randn(B, 4, 64, 64).to(self.accelerator.device, dtype=self.accelerator.unwrap_model(self.vae).dtype) 
        self.scheduler.set_timesteps(NUM_INFERENCE_STEPS)
        retval = patch_custom_attention(self.accelerator.unwrap_model(self.unet)) 

        # The hardcoded bboxes section is removed. `all_bboxes` is now used.

        for t_idx, t in enumerate(self.scheduler.timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)

            concat_encoder_states = torch.cat([uncond_encoder_states.repeat(B, 1, 1), encoder_states], dim=0)
            encoder_states_dict = {
                "encoder_hidden_states": concat_encoder_states,
                "attn_assignments": all_assignments,
                "args": self.args,
            }
            if self.replace_attn is not None:
                encoder_states_dict[self.replace_attn] = True

            encoder_states_dict["azimuths"] = all_azimuths
            encoder_states_dict["bboxes"] = all_bboxes # Use the dynamically provided bboxes

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=encoder_states_dict).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        images = self.accelerator.unwrap_model(self.vae).decode(latents.to(self.accelerator.device, dtype=self.accelerator.unwrap_model(self.vae).dtype)).sample

        save_path_global = osp.join(self.tmp_dir)
        os.makedirs(save_path_global, exist_ok=True)

        for idx, image in enumerate(images):
            image = (image / 2 + 0.5).clamp(0, 1).squeeze()
            image = (image * 255).to(torch.uint8)
            image = image.cpu().numpy().astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))
            image = np.ascontiguousarray(image)
            img_dim = image.shape[0]

            # `all_bboxes` contains `uncond_bboxes` (B items) then `cond_bboxes` (B items)
            # `all_bboxes[B + idx]` gives the list of bboxes for the current conditional image
            current_image_bboxes = all_bboxes[B + idx] # This is a list of bbox tensors for the current image

            colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)] # Some colors for bboxes
            for subject_bbox_idx, bbox_tensor in enumerate(current_image_bboxes):
                # bbox_tensor should be on CPU for numpy conversion if it was on GPU
                bbox_coords = bbox_tensor.cpu().numpy() 
                pt1 = (int(bbox_coords[0] * img_dim), int(bbox_coords[1] * img_dim))
                pt2 = (int(bbox_coords[2] * img_dim), int(bbox_coords[3] * img_dim))
                color = colors[subject_bbox_idx % len(colors)]
                cv2.rectangle(image, pt1, pt2, color, 2)

            image = Image.fromarray(image)
            image.save(save_paths[idx])


    def __call__(self, seed, gif_path, prompt, all_subjects_data, args):
        self.args = args
        self.replace_attn = args["replace_attn_maps"]

        normalize_merged_embedding = args["normalize_merged_embedding"] if "normalize_merged_embedding" in args.keys() else False

        self.accelerator.wait_for_everyone()
        if osp.exists(self.tmp_dir) and self.accelerator.is_main_process:
            shutil.rmtree(f"{self.tmp_dir}")
        self.accelerator.wait_for_everyone()

        with torch.no_grad():
            self.seed = seed
            self.gif_path = gif_path
            all_encoder_states = []
            all_save_paths = []
            all_attn_assignments = []
            all_track_ids = []
            all_interesting_token_strs = []
            all_azimuths = [] # List of lists: [[az_subj1_frame1, az_subj2_frame1], [az_subj1_frame2, az_subj2_frame2], ...]
            all_subjects = [] # List of lists: [[subj1_name_frame1, subj2_name_frame1], ...]
            all_bboxes_for_dataset = [] # List of lists of tensors: [[[bbox_s1_f1_coords], [bbox_s2_f1_coords]], [[bbox_s1_f2_coords], [bbox_s2_f2_coords]], ...]

            for gif_subject_data in all_subjects_data:
                subjects = []
                for subject_data in gif_subject_data:
                    subjects.append("_".join(subject_data["subject"].split()))
                subjects_string = "__".join(subjects)

                unique_strings = []
                for asset_idx in range(len(gif_subject_data)):
                    unique_string_subject = ""
                    for token_idx in range(self.merged_emb_dim // 1024):
                        unique_string_subject = unique_string_subject + f"{UNIQUE_TOKENS[f'{asset_idx}_{token_idx}']} "
                    unique_string_subject = unique_string_subject.strip()
                    unique_strings.append(unique_string_subject)

                n_samples = len(gif_subject_data[0]["normalized_azimuths"]) - 1

                mlp_embs_video = [] # This seems to collect embeddings for all subjects over all frames for ONE gif_subject_data
                # Let's trace mlp_embs_video:
                # Outer list: frames
                # Inner list: subjects
                # Element: mlp_emb for that subject for that frame

                # Temporary storage for one video/scene before appending to all_azimuths, all_subjects, all_bboxes_for_dataset
                video_mlp_embs_per_frame_per_subject = []

                for sample_idx in range(n_samples): # Iterates through frames for the current scene
                    mlp_embs_frame = []
                    azimuths_frame = []
                    subjects_frame = []
                    bboxes_frame_for_this_sample = [] # BBoxes for all subjects in this specific frame

                    for subject_idx, subject_data in enumerate(gif_subject_data): # Iterates through subjects in the current scene
                        azimuths_frame.append(2 * math.pi * subject_data["normalized_azimuths"][sample_idx])
                        subjects_frame.append(subject_data["subject"])
                        
                        # Get bbox for current subject, current frame
                        # Ensure "bboxes" key exists and has enough entries
                        if "bboxes" not in subject_data or len(subject_data["bboxes"]) <= sample_idx:
                            raise ValueError(f"Bounding box data missing or insufficient for subject '{subject_data['subject']}' at sample_idx {sample_idx}")
                        bbox_for_subject_frame = subject_data["bboxes"][sample_idx]
                        bboxes_frame_for_this_sample.append(torch.tensor(bbox_for_subject_frame, dtype=torch.float32)) # Store as tensor

                        normalized_azimuth = subject_data["normalized_azimuths"][sample_idx]
                        pose_input = torch.tensor([normalized_azimuth]).float().to(self.accelerator.device)
                        mlp_emb = self.merger(pose_input)
                        assert mlp_emb.shape == (1, self.merged_emb_dim)
                        mlp_emb = mlp_emb.squeeze()
                        mlp_embs_frame.append(mlp_emb)
                    
                    all_azimuths.append(azimuths_frame)
                    all_subjects.append(subjects_frame)
                    all_bboxes_for_dataset.append(bboxes_frame_for_this_sample) # Append bboxes for this frame to the main dataset list

                    video_mlp_embs_per_frame_per_subject.append(torch.stack(mlp_embs_frame, 0)) # mlp_embs for all subjects in this frame

                # video_mlp_embs_per_frame_per_subject is now a list (frames) of tensors (subjects, emb_dim)
                merged_embs_video = torch.stack(video_mlp_embs_per_frame_per_subject, 0) # (n_samples, n_subjects, merged_emb_dim)

                placeholder_text = "a SUBJECT0 "
                for asset_idx in range(1, len(gif_subject_data)):
                    placeholder_text = placeholder_text + f"and a SUBJECT{asset_idx} "
                placeholder_text = placeholder_text.strip()
                assert prompt.find("PLACEHOLDER") != -1
                template_prompt = prompt.replace("PLACEHOLDER", placeholder_text)

                for asset_idx, subject_data in enumerate(gif_subject_data):
                    assert template_prompt.find(f"SUBJECT{asset_idx}") != -1
                    template_prompt = template_prompt.replace(f"SUBJECT{asset_idx}", f"{unique_strings[asset_idx]} {subject_data['subject']}")

                prompt_ids = self.tokenizer(
                    template_prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(self.accelerator.device)

                track_ids_template = []
                interesting_token_strs_template = []
                for token_pos, token in enumerate(prompt_ids[0]):
                    if token in TOKEN2ID.values():
                        track_ids_template.append(token_pos)
                        interesting_token_strs_template.append(self.tokenizer.decode(token))
                
                # Construct attn_assignments template once per prompt, as it depends on token positions
                attn_assignments_template = {}
                for asset_idx in range(len(gif_subject_data)): # Iterate over subjects in the scene
                    for token_idx in range(self.merged_emb_dim // 1024): # Iterate over unique tokens per subject
                        unique_token_str_key = f"{asset_idx}_{token_idx}"
                        unique_token_literal = UNIQUE_TOKENS[unique_token_str_key]
                        unique_token_id_val = TOKEN2ID[unique_token_literal]
                        
                        # Find position of this unique_token_id_val in prompt_ids
                        # Ensure prompt_ids is 1D before converting to list
                        prompt_ids_list = prompt_ids.squeeze().tolist()
                        if unique_token_id_val not in prompt_ids_list:
                            # This can happen if tokenizer merges tokens, or if unique token isn't actually in this prompt
                            # For this model, we assume unique tokens ARE in the prompt
                            raise ValueError(f"Unique token {unique_token_literal} (ID: {unique_token_id_val}) not found in prompt_ids for prompt: {template_prompt}")
                        
                        unique_token_idx_in_prompt = prompt_ids_list.index(unique_token_id_val)
                        # The assignment logic: unique_token_idx -> unique_token_idx + (num_parts_per_emb - part_offset)
                        # For example, if merged_emb_dim is 2048 (2*1024), then two unique tokens make one concept.
                        # token_idx = 0 (first part): assign to unique_token_idx_in_prompt + 2 - 0 = unique_token_idx_in_prompt + 2
                        # token_idx = 1 (second part): assign to unique_token_idx_in_prompt + 2 - 1 = unique_token_idx_in_prompt + 1
                        # This seems to assign to positions *after* the unique token itself.
                        # Let's re-verify the original logic if possible.
                        # The original code had: attn_assignments[unique_token_idx] = unique_token_idx + self.merged_emb_dim // 1024 - token_idx
                        # If self.merged_emb_dim // 1024 = N_parts.
                        # token_idx=0: assign to pos + N_parts
                        # token_idx=1: assign to pos + N_parts - 1
                        # ...
                        # token_idx=N_parts-1: assign to pos + 1
                        # This maps the unique token placeholders to subsequent token positions for attention purposes.
                        attn_assignments_template[unique_token_idx_in_prompt] = unique_token_idx_in_prompt + (self.merged_emb_dim // 1024) - token_idx

                for sample_idx in range(n_samples): # This loop generates one data point for the dataset
                    for asset_idx, subject_data in enumerate(gif_subject_data):
                        subject = subject_data["subject"]
                        for token_idx in range(self.merged_emb_dim // 1024):
                            # merged_embs_video shape: (n_samples, n_subjects, merged_emb_dim)
                            replacement_emb = merged_embs_video[sample_idx, asset_idx] # Correct indexing
                            
                            if normalize_merged_embedding:
                                replacement_emb_norm = torch.linalg.norm(replacement_emb)
                                # Check if TOKEN2ID[subject] is valid; some subjects might not be base tokens
                                if subject in TOKEN2ID:
                                    org_emb_norm = torch.linalg.norm(self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[TOKEN2ID[subject]])
                                    if replacement_emb_norm > 1e-6: # Avoid division by zero
                                        replacement_emb = replacement_emb * org_emb_norm / replacement_emb_norm
                                    else: # Handle zero norm replacement embedding if necessary
                                        replacement_emb = replacement_emb # Or set to a default, or error
                                # else: If subject not in TOKEN2ID, cannot normalize against its original embedding norm.
                                # This case should be handled based on model design. For now, skip normalization if base token unknown.


                            self.accelerator.unwrap_model(self.text_encoder).get_input_embeddings().weight[TOKEN2ID[UNIQUE_TOKENS[f"{asset_idx}_{token_idx}"]]] = replacement_emb
                    
                    text_embeddings = self.text_encoder(prompt_ids)[0].squeeze() # Squeeze to (77, 1024)
                    all_encoder_states.append(text_embeddings)
                    all_save_paths.append(osp.join(self.tmp_dir, subjects_string, f"{str(sample_idx).zfill(3)}.jpg"))
                    
                    all_track_ids.append(track_ids_template) # Use the template
                    all_interesting_token_strs.append(interesting_token_strs_template) # Use the template
                    all_attn_assignments.append(attn_assignments_template) # Use the template for this frame

            self.accelerator.wait_for_everyone()
            self.accelerator.print(f"every thread finished generating the encoder hidden states...")

            if self.accelerator.is_main_process:
                for save_path in all_save_paths:
                    os.makedirs(osp.dirname(save_path), exist_ok=True)
            self.accelerator.wait_for_everyone()

            dataset = EncoderStatesDataset(all_encoder_states, all_save_paths, all_attn_assignments, all_track_ids, all_interesting_token_strs, all_azimuths, all_subjects, all_bboxes_for_dataset) # Pass bboxes

            dataloader = DataLoader(dataset, batch_size=self.bs, collate_fn=collate_fn)
            dataloader = self.accelerator.prepare(dataloader)

            self.accelerator.wait_for_everyone()
            self.accelerator.print(f"every thread finished preparing their dataloaders...")
            self.accelerator.print(f"starting generation...")
            for batch_idx, batch in enumerate(dataloader):
                self.generate_images_in_a_batch_and_save_them(batch, batch_idx)

            self.accelerator.wait_for_everyone()
            self.accelerator.print(f"every thread finished their generation, now collecting them to form a gif...")
            if self.accelerator.is_main_process:
                collect_generated_images(self.tmp_dir, prompt, self.gif_path)
            self.accelerator.wait_for_everyone()


if __name__ == "__main__":
    with torch.no_grad():
        args_path = osp.join(f"../ckpts/multiobject/", f"__{WHICH_MODEL}", f"args.pkl")
        assert osp.exists(args_path), f"{args_path = }"
        with open(args_path, "rb") as f:
            args = pickle.load(f)

        pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
        merger = CompassEncoder(args['merged_emb_dim'])

        training_state_path = osp.join(f"../ckpts/multiobject/", f"__{WHICH_MODEL}", f"training_state_{WHICH_STEP}.pth")
        assert osp.exists(training_state_path), f"{training_state_path = }"
        training_state = torch.load(training_state_path)

        if args['train_unet']:
            with torch.no_grad():
                _, _ = inject_trainable_lora(pipeline.unet, r=args['lora_rank'])
            unet_state_dict = pipeline.unet.state_dict()
            pretrained_unet_state_dict = training_state["unet"]["lora"]
            for name, param in unet_state_dict.items():
                if name.find("lora") == -1:
                    continue
                assert name in pretrained_unet_state_dict.keys()
                unet_state_dict[name] = pretrained_unet_state_dict[name]
            pipeline.unet.load_state_dict(unet_state_dict)

        merger.load_state_dict(training_state["merger"]["model"], strict=False)

        accelerator = Accelerator()
        replace_attn = args['replace_attn_maps']
        infer = Infer(args['merged_emb_dim'], accelerator, pipeline.unet, pipeline.scheduler, pipeline.vae, pipeline.text_encoder, pipeline.tokenizer, merger, f"tmp_{WHICH_MODEL}_{WHICH_STEP}_{KEYWORD}", bs=4) 

        bbox_subject1_static = [[0.00, 0.60, 0.45, 1.00]] * (NUM_SAMPLES + 1) # x_min, y_min, x_max, y_max
        bbox_subject2_static = [[0.55, 0.55, 1.00, 1.00]] * (NUM_SAMPLES + 1)

        subjects = [
            [
                {
                    "subject": "sedan",
                    "normalized_azimuths": np.linspace(0, 1, NUM_SAMPLES + 1),
                    "bboxes": bbox_subject1_static # Use static bboxes for sedan
                    # "bboxes": bbox_subject1_dynamic # Or use dynamic bboxes
                },
                {
                    "subject": "ferrari",
                    "normalized_azimuths": 1 - np.linspace(0, 1, NUM_SAMPLES + 1),
                    "bboxes": bbox_subject2_static # Use static bboxes for suv
                }
            ],
        ]
        prompts = [
            "a photo of PLACEHOLDER in a backyard of a bungalow on a bright sunny afternoon, high quality, sharp, best quality, high resolution",
        ]
        for prompt in prompts:
            # Make sure the GIF path is unique if running multiple times or with different subject sets for the same prompt
            # For simplicity, current path doesn't distinguish between different `subjects` inputs for the same prompt.
            gif_name_prompt_part = '_'.join(prompt.split()[:5]) # Use first 5 words of prompt for filename
            output_gif_path = osp.join(f"latents{WHICH_LATENTS}_inference_results", f"__{WHICH_MODEL}_{WHICH_STEP}_{MAX_SUBJECTS_PER_EXAMPLE}_{replace_attn}_{KEYWORD}", f"{gif_name_prompt_part}.gif")
            os.makedirs(osp.dirname(output_gif_path), exist_ok=True) # Ensure directory exists
            
            infer(None, output_gif_path, prompt, subjects, args)