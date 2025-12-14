from multiprocessing import Process
import os
import random
import torch
import pandas as pd
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


def worker(gpu_id, video_indices):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    experiment = 5
    epoch = 10

    print(f"[GPU {gpu_id}] loading model...")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
        ],
    )

    lora_path = f"models/train/Experiment{experiment:02d}/epoch-{epoch}.safetensors"
    pipe.load_lora(pipe.dit, lora_path, alpha=1)
    # pipe.enable_vram_management()

    df = pd.read_csv("data/nuscenes_test/metadata_test.csv")

    base_out_dir = f"evaluation/fix_frame_multi_prompts"
    os.makedirs(base_out_dir, exist_ok=True)

    for idx in video_indices:
        row = df.loc[idx]
        input_video_path = os.path.join("data/nuscenes_test", row["video"])
        video_name = row["video"].split("/")[1]

        input_image = VideoData(input_video_path, height=480, width=832)[0]

        all_indices = list(range(len(df)))
        all_indices.remove(idx)
        sampled_indices = random.sample(all_indices, 8)

        out_root = os.path.join(base_out_dir, video_name)
        os.makedirs(out_root, exist_ok=True)

        print(f"[GPU {gpu_id}] video {video_name}")

        for p_i, p_idx in enumerate(sampled_indices):
            prompt = df.loc[p_idx, "prompt"]
            prompt_dir = os.path.join(out_root, prompt)
            os.makedirs(prompt_dir, exist_ok=True)

            video = pipe(
                prompt=prompt,
                input_image=input_image,
                num_frames=161,
                seed=1,
                tiled=True,
            )

            save_video(video, os.path.join(prompt_dir, "video.mp4"), fps=16, quality=5)

            frame_dir = os.path.join(prompt_dir, "frames")
            os.makedirs(frame_dir, exist_ok=True)
            for i, frame in enumerate(video):
                frame.save(os.path.join(frame_dir, f"{i:06d}.jpg"))

    print(f"[GPU {gpu_id}] done")

if __name__ == "__main__":
    # num_gpus = 8
    # df = pd.read_csv("data/nuscenes_test/metadata_test.csv")

    # indices = list(range(len(df)))
    # splits = [indices[i::num_gpus] for i in range(num_gpus)]

    num_gpus = 8
    NUM_FIRST_FRAMES = 50
    RANDOM_SEED = 42

    df = pd.read_csv("data/nuscenes_test/metadata_test.csv")

    all_indices = list(range(len(df)))

    random.seed(RANDOM_SEED)
    selected_firstframe_indices = random.sample(
        all_indices,
        NUM_FIRST_FRAMES
    )

    # 把 50 个首帧分给不同 GPU
    splits = [
        selected_firstframe_indices[i::num_gpus]
        for i in range(num_gpus)
    ]
    processes = []
    for gpu_id in range(num_gpus):
        if len(splits[gpu_id]) == 0:
            continue
        p = Process(target=worker, args=(gpu_id, splits[gpu_id]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

