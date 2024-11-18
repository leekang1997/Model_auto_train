import subprocess
import datetime

# 配置训练命令和日志文件路径
training_command = "llamafactory-cli train /home/likang/LLaMA-Factory/examples/train_lora/2024-11-12-0.5-qwen.yaml"
log_file_path = f"train_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

# 函数运行训练并记录日志
def run_training_with_logging():
    try:
        with open(log_file_path, "w") as log_file:
            print("开始运行训练任务...")
            # 使用subprocess运行训练命令
            process = subprocess.Popen(
                training_command,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
            process.communicate()  # 等待任务完成
            if process.returncode == 0:
                print(f"训练任务完成，日志已保存到 {log_file_path}")
            else:
                print(f"训练任务失败，请检查日志文件：{log_file_path}")
    except Exception as e:
        print(f"运行训练任务时发生错误: {e}")

# 运行训练脚本
if __name__ == "__main__":
    run_training_with_logging()
