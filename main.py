import os
import json
import subprocess
from datetime import datetime

# 插入数据集信息的函数
def insert_dataset_info(json_file_path, new_dataset):
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.update(new_dataset)
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("数据集信息已成功添加！")
        return list(new_dataset.keys())[0]  # 返回数据集名称
    except Exception as e:
        print(f"修改 JSON 文件时出错：{e}")
        return None

# YAML 文件生成函数
def generate_yaml(output_path, params):
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

### output
output_dir: {output_dir}
logging_steps: {logging_steps}
save_steps: {save_steps}
overwrite_output_dir: {overwrite_output_dir}

### train
per_device_train_batch_size: {train_batch_size}
gradient_accumulation_steps: {gradient_accumulation_steps}
learning_rate: {learning_rate}
num_train_epochs: {num_epochs}
"""
    content = template_content.format(**params)
    with open(output_path, "w") as yaml_file:
        yaml_file.write(content)
    print(f"YAML 文件已生成：{output_path}")

# 运行训练并记录日志
def run_training_with_logging(yaml_path, gpu_devices="4,5,6,7"):
    log_file_path = f"train_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    factory_dir = "/home/likang/LLaMA-Factory"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_devices  # 设置可用 GPU
    activate_and_run = (
        f"source /home/likang/anaconda3/bin/activate llama && llamafactory-cli train {yaml_path}"
    )
    try:
        with open(log_file_path, "w") as log_file:
            print(f"切换到目录 {factory_dir} 并开始运行训练任务...")
            process = subprocess.Popen(
                activate_and_run,
                shell=True,
                cwd=factory_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                executable="/bin/bash",
                env=env
            )
            process.communicate()
            if process.returncode == 0:
                print(f"训练任务完成，日志已保存到 {log_file_path}")
            else:
                print(f"训练任务失败，请检查日志文件：{log_file_path}")
    except Exception as e:
        print(f"运行训练任务时发生错误: {e}")

# 主程序
if __name__ == "__main__":
    # 用户可配置的参数
    dataset_info_path = "/home/likang/LLaMA-Factory/data/dataset_info.json"
    dataset_file_path = "/home/likang/angang_data_contronak/DataContronal/2024-11-19-new-json_divided/flattened_dataset.json"
    model_path = "/home/likang/.cache/modelscope/hub/Qwen/Qwen2___5-0___5B-Instruct/"
    save_dir = "/home/likang/LLaMA-Factory/saves"
    yaml_dir = "/home/likang/LLaMA-Factory/examples/train_lora"
    cutoff_len = 10000  # 最大输入长度
    gpu_devices = "4,5,6,7"  # 可用 GPU

    # 数据集名称和配置
    dataset_name = "2024-11-19-angang_full"
    new_dataset = {
        dataset_name: {
            "file_name": dataset_file_path,
            "columns": {
                "prompt": "prompt",
                "query": "input",
                "response": "output"
            }
        }
    }

    # 插入新的数据集信息
    dataset_name = insert_dataset_info(dataset_info_path, new_dataset)
    if dataset_name:
        # 动态生成训练参数
        params = {
            "model_path": model_path,
            "stage": "sft",
            "do_train": "true",
            "finetuning_type": "lora",
            "lora_target": "all",
            "deepspeed_config": "examples/deepspeed/ds_z3_config.json",
            "dataset_name": dataset_name,
            "template_name": "qwen",
            "cutoff_len": cutoff_len,
            "output_dir": os.path.join(save_dir, dataset_name),
            "logging_steps": 10,
            "save_steps": 500,
            "overwrite_output_dir": "true",
            "train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "learning_rate": 1.0e-4,
            "num_epochs": 5.0,
        }

        # 动态生成 YAML 文件路径
        yaml_filename = f"{dataset_name}-{params['finetuning_type']}-qwen.yaml"
        yaml_path = os.path.join(yaml_dir, yaml_filename)

        # 生成 YAML 文件
        generate_yaml(yaml_path, params)

        # 使用生成的 YAML 文件运行训练
        run_training_with_logging(yaml_path, gpu_devices=gpu_devices)
