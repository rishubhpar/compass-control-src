export RUN_NAME="code_release2" 
# export RUN_NAME="debug" 

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export INSTANCE_DIR_1SUBJECT="../compass_dataset/ref_imgs_1subject"  
export INSTANCE_DIR_2SUBJECTS="../compass_dataset/ref_imgs_2subjects" 
export CONTROLNET_DIR_1SUBJECT="../compass_dataset/controlnet_imgs_1subject"
export CONTROLNET_DIR_2SUBJECTS="../compass_dataset/controlnet_imgs_2subjects"
export OUTPUT_DIR="../ckpts/multiobject/"
export CLASS_DATA_DIR="../compass_dataset/prior_imgs" 
export CONTROLNET_PROMPTS_FILE="../prompts/prompts_2410.txt" 
export VIS_DIR="../multiobject/"  


accelerate launch --config_file accelerate_config.yaml train.py \
  --train_unet="Y" \
  --train_text_encoder="N" \
  --use_controlnet_images="Y" \
  --use_ref_images="Y" \
  --learning_rate=1e-4 \
  --learning_rate_merger=1e-4 \
  --replace_attn_maps="class2special_soft" \
  --color_jitter="Y" \
  --center_crop="N" \
  --lr_warmup_steps=0 \
  --normalize_merged_embedding="N" \
  --merged_emb_dim=1024 \
  --with_prior_preservation="Y" \
  --root_data_dir=$ROOT_DATA_DIR \
  --controlnet_prompts_file=$CONTROLNET_PROMPTS_FILE \
  --stage1_steps=5000 \
  --stage2_steps=15000 \
  --resolution=512 \
  --train_batch_size=2 \
  --prior_loss_weight=1.0 \
  --gradient_accumulation_steps=1 \
  --run_name="$RUN_NAME" \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --controlnet_data_dir_2subjects=$CONTROLNET_DIR_2SUBJECTS \
  --controlnet_data_dir_1subject=$CONTROLNET_DIR_1SUBJECT \
  --instance_data_dir_1subject=$INSTANCE_DIR_1SUBJECT \
  --instance_data_dir_2subjects=$INSTANCE_DIR_2SUBJECTS \
  --output_dir=$OUTPUT_DIR \
  --vis_dir=$VIS_DIR \
  --class_data_dir=$CLASS_DATA_DIR 


  # in case you want to resume training, uncomment the following line and specify the path to the training state file  
  # --resume_training_state="../ckpts/multiobject/__class2special_detached__noloc_cond/training_state_390000.pth" \