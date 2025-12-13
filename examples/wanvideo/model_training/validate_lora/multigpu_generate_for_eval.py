import os
import torch
import pandas as pd
from multiprocessing import Process
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig


def run_single_task(gpu_id, experiment, epoch):
    print(f"[GPU {gpu_id}] Running Experiment{experiment:02d} | epoch {epoch}")

    # 绑定 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 加载基础模型
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
        ],
    )

    # 加载 lora
    lora_path = f"models/train/Experiment{experiment:02d}/epoch-{epoch}.safetensors"
    if not os.path.exists(lora_path):
        print(f"[GPU {gpu_id}] WARNING: LoRA not found -> {lora_path}")
        return

    pipe.load_lora(pipe.dit, lora_path, alpha=1)
    pipe.enable_vram_management()

    # 输出目录
    out_dir = f"evaluation/Experiment{experiment:02d}/epoch_{epoch}"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv("data/nuscenes_test/metadata_test.csv")

    for idx in range(len(df)):
        prompt = df.loc[idx, "prompt"]
        input_video_path = "data/nuscenes_test/" + df.loc[idx, "video"]
        video_name = df.loc[idx, "video"].split("/")[1]

        input_image = VideoData(input_video_path, height=480, width=832)[0]

        video = pipe(
            prompt=prompt,
            negative_prompt=(
                "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
            ),
            input_image=input_image,
            num_frames=161,
            seed=1,
            tiled=True,
        )

        save_video(video, f"{out_dir}/{video_name}", fps=16, quality=5)

    print(f"[GPU {gpu_id}] Finished Experiment{experiment:02d} | epoch {epoch}")


if __name__ == "__main__":
    # Experiments 1-7
    experiments = list(range(1, 8))

    # 每个实验默认都跑 epoch 20
    default_epoch = 20

    # Experiment 5 的额外 epoch
    extra_epochs_for_exp5 = [80]

    # 构建总任务列表
    tasks = []

    for exp in experiments:
        tasks.append((exp, default_epoch))  # 每个实验跑 epoch 20
        if exp == 5:
            for ep in extra_epochs_for_exp5:
                tasks.append((exp, ep))      # Experiment 5 额外跑 epoch 80

    print("Tasks to run:", tasks)

    # 设置 GPU 数量
    num_gpus = 8  # 可根据你机器调整

    processes = []
    for task_id, (exp, ep) in enumerate(tasks):
        gpu_id = task_id % num_gpus
        p = Process(target=run_single_task, args=(gpu_id, exp, ep))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


