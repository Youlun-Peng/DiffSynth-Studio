accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/nuscenes_dataset/ \
  --dataset_metadata_path data/nuscenes_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 117 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 7e-5 \
  --num_epochs 100 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Experiment07" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 64 \
  --extra_inputs "input_image" \
  --data_file_keys "video" 
  