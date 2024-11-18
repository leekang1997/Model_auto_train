import os
import subprocess
import datetime

# YAML模板内容
template_content = """### model
model_name_or_path: {model_path}

### method
stage: {stage}
do_train: {do_train}
finetuning_type: {finetuning_type}
lora_target: {lora_target}
deepspeed: {deepspeed_config}

### dataset
dataset: {dataset_name}
template: {template_name}
cutoff_len: {cutoff_len}
max_samples: {max_samples}
overwrite_cache: {overwrite_cache}
preprocessing_num_workers: {num_workers}

### output
output_dir: {output_dir}
logging_steps: {logging_steps}
save_steps: {save_steps}
plot_loss: {plot_loss}
overwrite_output_dir: {overwrite_output_dir}

### train
per_device_train_batch_size: {train_batch_size}
gradient_accumulation_steps: {gradient_accumulation_steps}
learning_rate: {learning_rate}
num_train_epochs: {num_epochs}
lr_scheduler_type: {lr_scheduler}
warmup_ratio: {warmup_ratio}
bf16: {bf16}
ddp_timeout: {ddp_timeout}

### eval
val_size: {val_size}
per_device_eval_batch_size: {eval_batch_size}
eval_strategy: {eval_strategy}
eval_steps: {eval_steps}
"""

# 参数配置
params = {
    "model_path": "/home/likang/.cache/modelscope/hub/Qwen/Qwen2___5-0___5B-Instruct/",
    "stage": "sft",
    "do_train": "true",
    "finetuning_type": "lora",
    "lora_target": "all",
    "deepspeed_config": "examples/deepspeed/ds_z3_config.json",
    "dataset_name": "2024-11-12-angang_fake_time_model",
    "template_name": "qwen",
    "cutoff_len": 20000,
    "max_samples": 1000000,
    "overwrite_cache": "true",
    "num_workers": 16,
    "output_dir": "/home/likang/LLaMA-Factory/saves/leekang/qwen2.5_0.5b_2024-11-12_lora_time_model",
    "logging_steps": 10,
    "save_steps": 500,
    "plot_loss": "true",
    "overwrite_output_dir": "true",
    "train_batch_size": 1,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1.0e-4,
    "num_epochs": 5.0,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.1,
    "bf16": "true",
    "ddp_timeout": 180000000,
    "val_size": 0.1,
    "eval_batch_size": 1,
    "eval_strategy": "steps",
    "eval_steps": 500,
}

# YAML 文件生成函数
def generate_yaml(output_path, params):
    content = template_content.format(**params)
    with open(output_path, "w") as yaml_file:
        yaml_file.write(content)
    print(f"YAML文件已生成：{output_path}")

# 运行训练并记录日志的函数
def run_training_with_logging(yaml_path):
    log_file_path = f"train_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    training_command = f"llamafactory-cli train {yaml_path}"

    try:
        with open(log_file_path, "w") as log_file:
            print("开始运行训练任务...")
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

# 启动 VLLM 服务的函数
def start_vllm_service(model_path, model_name, port, tensor_parallel_size):
    vllm_command = (
        f"export VLLM_WORKER_MULTIPROC_METHOD=spawn && "
        f"export CUDA_VISIBLE_DEVICES=3 && "
        f"python3 -m vllm.entrypoints.openai.api_server "
        f"--model {model_path} "
        f"--served-model-name {model_name} "
        f"--trust-remote-code "
        f"--port {port} "
        #这里可以去掉
        f"--tensor-parallel-size {tensor_parallel_size}"
    )
    log_file_path = f"vllm_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    try:
        with open(log_file_path, "w") as log_file:
            print("启动 VLLM 服务...")
            process = subprocess.Popen(
                vllm_command,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
            process.communicate()  # 等待任务完成
            if process.returncode == 0:
                print(f"VLLM 服务启动成功，日志已保存到 {log_file_path}")
            else:
                print(f"VLLM 服务启动失败，请检查日志文件：{log_file_path}")
    except Exception as e:
        print(f"启动 VLLM 服务时发生错误: {e}")

# 主程序
if __name__ == "__main__":
    # 动态生成 YAML 文件路径
    yaml_filename = f"2024-11-12-{params['finetuning_type']}-qwen.yaml"
    yaml_path = os.path.join("/home/likang/LLaMA-Factory/examples/train_lora", yaml_filename)

    # 生成 YAML 文件
    generate_yaml(yaml_path, params)

    # 使用生成的 YAML 文件运行训练
    run_training_with_logging(yaml_path)

    # 在训练完成后自动启动 VLLM 服务
    start_vllm_service(
        model_path=params["model_path"],
        model_name="Qwen2-7B-Instruct",
        port=8009,
        tensor_parallel_size=8
    )
