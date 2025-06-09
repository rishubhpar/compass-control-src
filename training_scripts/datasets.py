import torch
import torch.nn as nn 
from torch.utils.data import Dataset
from torchvision import transforms

from pathlib import Path
from typing import Optional
import random
import re
from PIL import Image 
import glob
import sys 

import os 
import os.path as osp 

from infer import UNIQUE_TOKENS 

import numpy as np 
import pickle 

import copy 

MAX_SUBJECTS_PER_EXAMPLE = 2  
RAW_IMG_SIZE = 1024 

class CompassControlDataset(Dataset): 
    def pose_bin_to_pose_range(self, pose_bin): 
        # returns the range of poses that correspond to the given pose bin in radians 
        pose_range = (pose_bin * 2 * np.pi / self.num_pose_bins, (pose_bin + 1) * 2 * np.pi / self.num_pose_bins) 
        return pose_range 


    def pose_to_pose_bin(self, pose):  
        # returns the pose bin that corresponds to the given pose in radians 
        pose_bin = int(pose * self.num_pose_bins / (2 * np.pi)) 
        return pose_bin 


    def __init__(
        self, 
        args, 
        tokenizer, 
        ref_imgs_dirs: list, 
        controlnet_imgs_dirs: list, 
        num_steps, 
        gpu_idx: int, 
        num_pose_bins=10, 
        single_subject_augmentation_factor=1.0, 
    ): 
        self.args = args 

        # for this dataset, both controlnet and reference images must be used 
        assert self.args.use_controlnet_images == True 
        assert self.args.use_ref_images  == True 

        self.subjects = self.args.subjects 
        print(f"preparing dataset on gpu {gpu_idx}, {self.subjects = }")
        self.tokenizer = tokenizer 
        self.num_steps = num_steps 
        img_transforms = []

        self.ref_imgs_dirs = ref_imgs_dirs  
        self.controlnet_imgs_dirs = controlnet_imgs_dirs 

        if args.resize:
            img_transforms.append(
                transforms.Resize(
                    args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
                )
            )
        if args.center_crop:
            img_transforms.append(transforms.CenterCrop(args.resolution)) 
        if args.color_jitter:
            img_transforms.append(transforms.ColorJitter(0.2, 0.1))
        # if args.h_flip:
        #     img_transforms.append(transforms.RandomHorizontalFlip())

        self.image_transforms = transforms.Compose(
            [*img_transforms, transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

        self.num_pose_bins = num_pose_bins 

        self.single_subject_augmentation_factor = single_subject_augmentation_factor 

        self.subjects_to_idx = {subject: idx for idx, subject in enumerate(self.subjects)} 

        # creating a list of subjects pose-bins combinations that will be used for training 
        # self.subjects_poses = []  

        # # the double subject cases 
        # double_subject_poses = [] 
        # for subject1 in self.subjects: 
        #     for subject2 in self.subjects: 
        #         for pose_bin1 in range(self.num_pose_bins): 
        #             for pose_bin2 in range(self.num_pose_bins): 
        #                 double_subject_poses.append((subject1, subject2, pose_bin1, pose_bin2))

        # # the single subject cases
        # single_subject_poses = [] 
        # for subject in self.subjects: 
        #     for pose_bin in range(self.num_pose_bins): 
        #         single_subject_poses.append((subject, pose_bin)) 

        # self.subjects_poses = double_subject_poses + single_subject_poses * int(self.single_subject_augmentation_factor) 

        # # a storage for the available images for each subject pose combination 
        self.available_images = {} 
        # for subject_pose in self.subjects_poses: 
        #     self.available_images[subject_pose] = [] 

        self.class_prompt_to_ids = {} 
        for subject in self.subjects: 
            class_prompt = f"a photo of a {subject}" 
            class_prompt_ids = self.tokenizer( 
                class_prompt, 
                padding="max_length", 
                truncation=True, 
                max_length=self.tokenizer.model_max_length,  
                return_tensors="pt", 
            ).input_ids[0]  
            self.class_prompt_to_ids[class_prompt] = class_prompt_ids 


        all_dirs = controlnet_imgs_dirs + ref_imgs_dirs
        for imgs_dir_idx, imgs_dir in enumerate(all_dirs): 
            for subjects_string in os.listdir(imgs_dir): 
                subjects_path = osp.join(imgs_dir, subjects_string) 
                img_names = os.listdir(subjects_path) 
                img_names = [img_name for img_name in img_names if img_name.find("jpg") != -1] 

                if "prompt" in img_names[0]: 
                    pkl_names = ["__".join(img_name.split("__")[:-2]) + "__.pkl" for img_name in img_names]  
                    pkl_paths = [osp.join(all_dirs[imgs_dir_idx + len(all_dirs) // 2], subjects_string, pkl_name) for pkl_name in pkl_names]   
                else: 
                    pkl_names = [img_name.replace("jpg", "pkl") for img_name in img_names] 
                    pkl_paths = [osp.join(subjects_path, pkl_name) for pkl_name in pkl_names] 
                for pkl_path in pkl_paths: 
                    assert osp.exists(pkl_path), f"{pkl_path = }, {imgs_dir = }, {img_names[0] = }" 

                all_pkl_data = [] 
                for pkl_path in pkl_paths: 
                    with open(pkl_path, "rb") as f: 
                        pkl_data = pickle.load(f) 
                    all_pkl_data.append(pkl_data) 

                img_paths = [osp.join(subjects_path, img_name) for img_name in img_names] 

                for img_path, pkl_data in zip(img_paths, all_pkl_data): 
                    img_name = osp.basename(img_path) 
                    subjects = subjects_string.split("__") 

                    azimuths = [] 
                    bboxes = [] 
                    pose_bins = [] 
                    for k in pkl_data.keys(): 
                        if pkl_data[k] == {}: 
                            continue 
                        azimuth = pkl_data[k]["a"] 
                        azimuths.append(azimuth) 

                        bbox = torch.tensor(pkl_data[k]["bbox"]) / RAW_IMG_SIZE  
                        bboxes.append(bbox) 

                        pose_bin = self.pose_to_pose_bin(azimuth)  
                        pose_bins.append(pose_bin) 

                    if "prompt" in img_path: 
                        # controlnet 
                        prompt_idx = int(img_name.split("__")[-2].replace("prompt", ""))  
                        prompt = self.args.controlnet_prompts[prompt_idx] 
                    else: 
                        prompt = f"a photo of PLACEHOLDER in a dark studio with white lights" 

                    replacement_str = "" 
                    replacement_str = replacement_str + f"a {UNIQUE_TOKENS['0_0']} {subjects[0]}" 
                    for asset_idx, subject in enumerate(subjects):  
                        if asset_idx == 0: 
                            continue 
                        replacement_str = replacement_str + f" and a {UNIQUE_TOKENS[f'{asset_idx}_0']} {subjects[asset_idx]}" 
                    prompt = prompt.replace(f"PLACEHOLDER", replacement_str)  

                    prompt_ids = self.tokenizer(
                        prompt, 
                        padding="max_length", 
                        truncation=True, 
                        max_length=self.tokenizer.model_max_length, 
                        return_tensors="pt", 
                    ).input_ids[0]  

                    assert len(pose_bins) == len(subjects) 

                    if (*subjects, *pose_bins) not in self.available_images.keys(): 
                        self.available_images[(*subjects, *pose_bins)] = [] 

                    self.available_images[(*subjects, *pose_bins)].append({"img_path": img_path, "prompt": prompt, "prompt_ids": prompt_ids, "bboxes": bboxes, "azimuths" : azimuths, "subjects": subjects})    

        self.next_idx = {} 
        for k in self.available_images.keys(): 
            if len(self.available_images[k]) == 0: 
                self.next_idx[k] = -1  
            else: 
                self.next_idx[k] = (gpu_idx + 1) % len(self.available_images[k])  


    def __getitem__(self, index): 
        subject_pose = list(self.available_images.keys())[index % len(self.available_images.keys())] 
        chosen_idx = copy.deepcopy(self.next_idx[subject_pose])  
        chosen_img_data = self.available_images[subject_pose][chosen_idx] 
        self.next_idx[subject_pose] = (self.next_idx[subject_pose] + 1) % len(self.available_images[subject_pose])  
        img_path = chosen_img_data["img_path"] 
        img = Image.open(img_path).convert("RGB") 
        img = self.image_transforms(img) 
        example = {} 
        example["img"] = img 
        if chosen_idx == 0: 
            example["controlnet"] = False  
        else: 
            example["controlnet"] = True  

        # double subject case 
        subjects = chosen_img_data["subjects"] 
        azimuths = chosen_img_data["azimuths"] 
        bboxes = chosen_img_data["bboxes"] 
        prompt = chosen_img_data["prompt"] 
        prompt_ids = chosen_img_data["prompt_ids"] 

        example["subjects"] = subjects  
        example["scalers"] = torch.tensor(azimuths)  
        example["prompt_ids"] = prompt_ids  
        example["prompt"] = prompt  
        example["2d_xs"] = [] 
        example["2d_ys"] = [] 
        for bbox in bboxes: 
            example["2d_xs"].append((bbox[0] + bbox[2]) / 2)  
            example["2d_ys"].append((bbox[1] + bbox[3]) / 2)  
        example["bboxes"] = bboxes  


        if self.args.with_prior_preservation: 
            random_subject_ = random.choice(subjects)  
            random_subject = " ".join(random_subject_.split("_")).strip()  
            example["prior_subject"] = random_subject 
            subject_class_imgs_path = osp.join(self.args.class_data_dir, random_subject_)   
            assert len(os.listdir(subject_class_imgs_path)) == 100, f"{len(os.listdir(subject_class_imgs_path)) = }"  
            n_class_images = len(os.listdir(subject_class_imgs_path)) 
            class_img_name = str(index % n_class_images).zfill(3) + ".jpg"  
            class_img_path = osp.join(subject_class_imgs_path, class_img_name) 
            assert osp.exists(class_img_path), f"{class_img_path = }"
            class_img = Image.open(class_img_path).convert("RGB")  
            example["class_img"] = self.image_transforms(class_img) 
            class_prompt = f"a photo of a {random_subject}" 
            example["class_prompt"] = class_prompt 
            example["class_prompt_ids"] = self.class_prompt_to_ids[class_prompt] 

        assert len(example["subjects"]) == len(example["scalers"]) 

        return example 


    def __len__(self): 
        return self.num_steps + 16 # + 16 is to ensure that the dataset is large enough to cover all steps in the training loop  