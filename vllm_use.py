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
        f"--gpu_memory_utilization 0.1"
        f"--trust-remote-code "
        f"--port {port}"
    )

    try:
        print(f"切换到目录 {factory_dir} 并启动 vllm API 服务器...")
        # 启动 API 服务并将输出日志同时保存和打印
        process = subprocess.Popen(
            f"{activate_and_run} | tee {log_file_path}",
            shell=True,
            cwd=factory_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            executable="/bin/bash",
            env=env,
            text=True  # 确保输出是以文本形式处理
        )

        # 实时打印输出到控制台
        for line in iter(process.stdout.readline, ''):
            print(line, end='')

        process.stdout.close()  # 关闭 stdout 以避免死锁
        process.wait()  # 等待进程结束

        if process.returncode == 0:
            print(f"vllm 服务器启动成功，日志已保存到 {log_file_path}")
        else:
            print(f"vllm 服务器启动失败，请检查日志文件：{log_file_path}")
    except Exception as e:
        print(f"启动 vllm 服务器时发生错误: {e}")


# 示例调用
start_vllm_server(
    model_path="/home/extra_space/likang_model/2024-12-6-angang_first_govern_data_2022701_20230131_export_tes/",
    model_name="Qwen2-0.5B-Instruct",
    port=8007, gpu_devices="4"
)
