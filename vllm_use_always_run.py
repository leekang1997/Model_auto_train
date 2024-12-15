import subprocess
import os
from datetime import datetime


def start_vllm_server(model_path, model_name, port=8009, gpu_devices="4"):
    log_file_path = f"vllm_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    factory_dir = "/home/likang"
    env = os.environ.copy()

    # 设置环境变量
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["CUDA_VISIBLE_DEVICES"] = gpu_devices  # 设置可用 GPU

    # 激活 vllm 环境并启动服务
    activate_and_run = (
        f"source /home/likang/anaconda3/bin/activate vllm && "
        f"python3 -m vllm.entrypoints.openai.api_server "
        f"--model {model_path} "
        f"--served-model-name {model_name} "
        #f"--gpu_memory_utilization 0.4"
        f"--trust-remote-code "
        f"--port {port}"
    )

    screen_name = f"vllm_api_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    try:
        print(f"切换到目录 {factory_dir} 并在新的 screen 会话中启动 vllm API 服务器...")
        with open(log_file_path, 'w') as log_file:
            # 使用 screen 启动 API 服务并将输出日志仅保存到文件
            process = subprocess.Popen(
                [
                    "screen",
                    "-dmS", screen_name,
                    "bash", "-c",
                    f"{activate_and_run} | tee {log_file_path}"
                ],
                cwd=factory_dir,
                stdout=log_file,
                stderr=log_file,
                env=env,
                text=True  # 确保输出是以文本形式处理
            )

        print(f"vllm 服务器正在启动，在 screen 会话 '{screen_name}' 中运行，日志将保存到 {log_file_path}")

        return screen_name  # 返回 screen 会话名称以便后续管理

    except Exception as e:
        print(f"启动 vllm 服务器时发生错误: {e}")
        return None


# 示例调用
screen_session = start_vllm_server(
    model_path="/home/likang/.cache/modelscope/hub/Qwen/Qwen2___5-Coder-7B-Instruct-GPTQ-Int4/",
    model_name="Qwen2-7B-Instruct_int4",
    port=8007, gpu_devices="0"
)

if screen_session is not None:
    print(f"vllm 服务器已启动，在 screen 会话 '{screen_session}' 中运行")