import subprocess
import time

# 配置服务启动命令
start_service_command = """
export CUDA_VISIBLE_DEVICES=3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python3 -m vllm.entrypoints.openai.api_server \
        --model /home/likang/.cache/modelscope/hub/qwen/Qwen2-7B-Instruct/ \
        --served-model-name Qwen2-7B-Instruct \
        --trust-remote-code \
        --tensor-parallel-size 8
"""

# 启动模型服务
def start_model_service():
    try:
        print("正在启动模型服务...")
        process = subprocess.Popen(
            start_service_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # 给服务一些时间启动
        time.sleep(10)  # 根据实际情况调整等待时间
        print("模型服务启动完成。如果服务失败，请检查输出日志。")
        return process
    except Exception as e:
        print(f"启动模型服务时出错: {e}")
        return None

# 主程序
if __name__ == "__main__":
    service_process = start_model_service()
    if service_process:
        print("模型服务运行中...")
