import os
import subprocess
import json
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


# 生成用于导出的 YAML 配置文件
def generate_export_yaml(export_yaml_path, model_path, adapter_path, export_dir):
    template_content = """### Note: DO NOT use quantized model or quantization_bit when merging lora adapters

### model
model_name_or_path: {model_path}
adapter_name_or_path: {adapter_path}
template: qwen
finetuning_type: lora

### export
export_dir: {export_dir}
export_size: 2
export_device: cpu
export_legacy_format: false
"""
    content = template_content.format(
        model_path=model_path,
        adapter_path=adapter_path,
        export_dir=export_dir
    )
    with open(export_yaml_path, "w") as yaml_file:
        yaml_file.write(content)
    print(f"导出 YAML 文件已生成：{export_yaml_path}")


# 执行导出命令
import subprocess
import os


# 导出 Lora 模型的函数
def export_lora_model(merge_yaml_path):
    log_file_path = f"export_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    factory_dir = "/home/likang/LLaMA-Factory"
    env = os.environ.copy()

    # 构造激活环境并导出 Lora 模型的命令
    activate_and_run = (
        f"source /home/likang/anaconda3/bin/activate llama && "
        f"llamafactory-cli export {merge_yaml_path}"
    )

    try:
        print(f"切换到目录 {factory_dir} 并开始执行导出 Lora 模型任务...")
        # 使用 subprocess.Popen 运行命令
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
            print(f"Lora 模型导出完成，日志已保存到 {log_file_path}")
        else:
            print(f"Lora 模型导出失败，请检查日志文件：{log_file_path}")
    except Exception as e:
        print(f"导出 Lora 模型时发生错误: {e}")


# 主程序
if __name__ == "__main__":
    # 配置路径
    model_path = "/home/likang/.cache/modelscope/hub/Qwen/Qwen2___5-0___5B-Instruct/"
    adapter_path = "/home/extra_space/likang_model/equip_2024_12_10/2024-12-13-angang_equipment/"
    export_dir = "/home/extra_space/likang_model/govern_divide_tes_train_2024_12_13/"

    # 生成导出 YAML 配置文件路径
    export_yaml_path = f"/home/likang/LLaMA-Factory/examples/merge_lora/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_merge_lora.yaml"

    # 生成导出 YAML 文件
    generate_export_yaml(export_yaml_path, model_path, adapter_path, export_dir)

    # 执行导出 LORA 模型
    export_lora_model(export_yaml_path)
