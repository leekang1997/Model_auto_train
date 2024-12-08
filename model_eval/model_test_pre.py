import torch
from torch.utils.data import DataLoader
import requests
import json
from model_talk_funtion.Chat_BigModel import chat_BigModel
import argparse
import re
import numpy as np
import pandas as pd
import time

# 32个
features_zh = ['给矿量瞬时值设定', '2#给矿皮带给矿量瞬时值', '一次2#旋流器流量', '3#集矿皮带电流反馈', '4#集矿皮带给定频率', '累计加球量', '一次2#旋流器电动阀阀位', '3#集矿皮带频率反馈', '一次1#旋流器旋流器沉砂补加水(备)', '一次1#旋流器开启台数', '旋给压力设定值', '2#给矿皮带给矿量累计值', '2#旋流器给矿泵箱泵箱液位', '4#集矿皮带频率反馈',
               '磨机浓度系数', '磨音', '一次1#旋流器旋流器沉砂补加水', '2#旋流器给矿泵箱补水阀位', '2#块度仪-12mm粒度', '4#集矿皮带电流反馈', '一次1#旋流器旋流器沉砂补加水(备)设定', '2#旋流器给矿泵箱出水浓度', '泵池液位设定', '2#旋流器给矿泵箱出水流量', '一次2#旋流器压力', '一次1#旋流器旋流器沉砂补加水设定', '累计加球重', '磨机电流', '运行时间', ' 2#球磨机矿浆箱液位检测LI203', '3#集矿皮带给定频率', '旋溢粒度-200目']

# 32个
features_zh_pre = ['给矿量瞬时值设定-pre', '2#给矿皮带给矿量瞬时值-pre', '一次2#旋流器流量-pre', '3#集矿皮带电流反馈-pre', '4#集矿皮带给定频率-pre', '累计加球量-pre', '一次2#旋流器电动阀阀位-pre', '3#集矿皮带频率反馈-pre', '一次1#旋流器旋流器沉砂补加水(备)-pre', '一次1#旋流器开启台数-pre', '旋给压力设定值-pre', '2#给矿皮带给矿量累计值-pre', '2#旋流器给矿泵箱泵箱液位-pre', '4#集矿皮带频率反馈-pre',
               '磨机浓度系数-pre', '磨音-pre', '一次1#旋流器旋流器沉砂补加水-pre', '2#旋流器给矿泵箱补水阀位-pre', '2#块度仪-12mm粒度-pre', '4#集矿皮带电流反馈-pre', '一次1#旋流器旋流器沉砂补加水(备)设定-pre', '2#旋流器给矿泵箱出水浓度-pre', '泵池液位设定-pre', '2#旋流器给矿泵箱出水流量-pre', '一次2#旋流器压力-pre', '一次1#旋流器旋流器沉砂补加水设定-pre', '累计加球重-pre', '磨机电流-pre', '运行时间-pre', ' 2#球磨机矿浆箱液位检测LI203-pre', '3#集矿皮带给定频率-pre', '旋溢粒度-200目-pre']


error_index = 0 # 在解析值时出现不匹配情况的次数

# 从output和response中获取各指标的值组成字典，如果有科学计数法的情况将其转换为正常表示
def get_dict(data):
    # 正则表达式模式，用于匹配标签和值
    pattern = r'([\u4e00-\u9fa5\w]+(?:[-#][\u4e00-\u9fa5\w]+)*(?:\([^\)]*\)(?:[-#][\u4e00-\u9fa5\w]+)*)*):\s*([-\d.]+(?:[eE][-+]?\d+\.?\d*)?)'

    # 使用正则表达式查找所有匹配项
    matches = re.findall(pattern, data)
    data_dict = {}
    global error_index
    # 将匹配的标签和值添加到字典中
    for match in matches[2:]:
        label, value = match
        # 将科学计数法转换为正常表示
        if 'e' in value or 'E' in value:
            value = format(float(value), '.10f')
        try:
            data_dict[label] = float(value)  # 将值转换为浮点数
        except Exception as e:
            data_dict[label] = 123456789
            error_index += 1

    return data_dict

def calculate_RMSE(data_truth, data_pre):
    # 初始化一个空列表来存储每个指标的差的平方
    squared_diffs = []

    # 遍历data1中的每个键和值
    for key in data_pre:
        # 获取data2中对应的真实值
        true_value = data_truth.get(key, None)
        if true_value is not None:
            # 计算差的平方并将其作为一个列表添加到二维数组中
            squared_diffs.append(((float(data_pre[key]) - true_value) ** 2))
    print("squared_diffs", squared_diffs)
    return squared_diffs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, default='/home/likang/angang_data_contronak/DataContronal/2024-12-1-eam_shockwave_divided/step1_each/training_data_20241206_095424_test.json')
    parser.add_argument('--output_predict', type=str, default='predict_wenzhen.xlsx')
    parser.add_argument('--output_diffs', type=str, default='diffs_wenzhen.xlsx')
    parser.add_argument('--model_name', type=str, default='Qwen2-0.5B-Instruct')
    args = parser.parse_args()

    # 读取 JSON 文件
    with open(args.test_data, 'r') as file:
        data = json.load(file)

    squared_diffs_list = []  # 用于存储每次循环得到的squared_diffs
    error_log = []
    data_predict = []
    data_true = []
    time_start = time.time()  # 记录开始时间

    for item in data:
        try:
            input_text = item['input']
            output_text = item['output']
            prompt_text = item['prompt']
            print("Input:", input_text)
            print("Output:", output_text)
            prompt_input = prompt_text + input_text

            data_truth = get_dict(output_text)
            print("data_truth: ", data_truth)
            data_list = list(data_truth.values())
            data_true.append(data_list)

            response = chat_BigModel(prompt_input, [], args.model_name)
            print("response:", response)

            data_pre = get_dict(response)
            print("data_pre: ", data_pre)
            values_list = list(data_pre.values())
            data_predict.append(values_list)

            squared_diffs = calculate_RMSE(data_truth, data_pre)
            squared_diffs_list.append(squared_diffs)  # 将每次循环得到的squared_diffs添加到列表中
            print("\n")
        except Exception as e:
            error_log.append(f"在执行值 {response} 时发生错误: {e}")

    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)

    # 将squared_diffs_list转换为DataFrame
    df_squared_diffs = pd.DataFrame(squared_diffs_list, columns=features_zh)
    # 保存DataFrame为Excel文件
    excel_filename = args.output_diffs
    df_squared_diffs.to_excel(excel_filename, index=False)
    print(f"Squared diffs saved to {excel_filename}")

    data_predict_list = pd.DataFrame(data_predict, columns=features_zh_pre)
    # 保存DataFrame为Excel文件
    excel_filename1 = args.output_predict
    data_predict_list.to_excel(excel_filename1, index=False)
    print(f"data_predict_list saved to {excel_filename1}")

    data_true_list = pd.DataFrame(data_true)

    # 使用 concat 函数合并两个 DataFrame
    combined_data = pd.concat([data_true_list, data_predict_list], axis=1)

    combined_data.to_excel('hebing.xlsx', index=False)
    print(f"data_predict_list saved to hebing.xlsx")

    # 查看合并后的数据
    print(combined_data.head())

    # 打印错误日志
    print("错误日志:")
    for error in error_log:
        print(error)

    print("error_index = ", error_index)