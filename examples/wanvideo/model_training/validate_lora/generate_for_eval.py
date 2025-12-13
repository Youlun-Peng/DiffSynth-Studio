import torch
from PIL import Image
from diffsynth import save_video, VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download

import pandas as pd
import os


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
    ],
)
pipe.load_lora(pipe.dit, "models/train/Experiment05/epoch-8.safetensors", alpha=1)
pipe.enable_vram_management()

df = pd.read_csv("data/nuscenes_test/metadata_test.csv")
for index in range(len(df)):
    prompt = df.loc[index, "prompt"]
    input_video_path = df.loc[index, "video"]
    video_name = input_video_path.split("/")[1]
    input_video_path = "data/nuscenes_test/" + input_video_path

    input_image = VideoData(input_video_path, height=480, width=832)[0]
    video = pipe(
        prompt=prompt,
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=input_image,
        num_frames=161,
        seed=1, tiled=True,
    )

    save_video(video, f"evaluation/experiment05/{video_name}", fps=16, quality=5)
