import pandas as pd
import numpy as np

# 计算平方差/绝对差/绝对差和原始差的比值
# # 读取Excel文件
file_path = 'E:/DeskTop/结果/时空11.20/predict_chazhi.xlsx'  # 替换为你的Excel文件路径
# file_path = 'E:/DeskTop/结果/11.6time_pre结果/data_predict.xlsx'
df = pd.read_excel(file_path)
index = 0
features_zh = ['振动最大值', '振动最小值', '峰峰值', '平均值', '标准差', '均方根值', '方根幅值', '绝对平均值', '偏度', '峭度', '峰值指标', '波形指标', '脉冲指标', '裕度指标',
               '频率平均值', '频率方差', '频率偏度', '频率峭度', '重心频率', '频率标准差', '频率均方根', '平均频率', '规则度', '变化参数', '八阶矩', '十六阶矩']

# 创建一个新的DataFrame来存储结果
new_df = pd.DataFrame()

for i in range(26):
    true_col = features_zh[index]
    pred_col = true_col + "-pre"

    # 计算差的平方，并插入到DataFrame中
    square_diff_col = true_col + "-平方差"
    square_diff = (df[true_col] - df[pred_col]) ** 2
    df.insert(loc=df.columns.get_loc(pred_col) + 1, column=square_diff_col, value=square_diff)

    # 计算差的绝对值，并插入到DataFrame中
    abs_diff_col = true_col + "-绝对差"
    abs_diff = (df[true_col] - df[pred_col]).abs()
    df.insert(loc=df.columns.get_loc(pred_col) + 2, column=abs_diff_col, value=abs_diff)

    # 计算差的绝对值比原始值，并插入到DataFrame中
    abs_diff_over_true_col = true_col + "-绝对差/原始值"
    # 避免除以零的错误
    abs_diff_over_true = ((df[true_col] - df[pred_col]).abs() / df[true_col]).abs()
    # 将结果中的inf替换为0
    abs_diff_over_true = abs_diff_over_true.replace([np.inf, -np.inf], 0)
    df.insert(loc=df.columns.get_loc(pred_col) + 3, column=abs_diff_over_true_col, value=abs_diff_over_true)

    index += 1



# 计算最终的四个评价指标
# 将结果写入新的Excel文件
output_path = 'output_file.xlsx'  # 输出文件的路径
df.to_excel(output_path, index=False)  # index=False表示不写入行索引


file_path = 'output_file.xlsx'
df = pd.read_excel(file_path)
index = 0
features_zh = ['振动最大值', '振动最小值', '峰峰值', '平均值', '标准差', '均方根值', '方根幅值', '绝对平均值', '偏度', '峭度', '峰值指标', '波形指标', '脉冲指标', '裕度指标',
               '频率平均值', '频率方差', '频率偏度', '频率峭度', '重心频率', '频率标准差', '频率均方根', '平均频率', '规则度', '变化参数', '八阶矩', '十六阶矩']
# 创建一个新的DataFrame来存储结果
results_df = pd.DataFrame(columns=['特征', 'MSE', 'RMSE', 'MAE', 'MAPE'])

for i in range(26):
    column_name1 = features_zh[index] + "-平方差"
    column_name2 = features_zh[index] + "-绝对差"
    column_name3 = features_zh[index] + "-绝对差/原始值"

    MSE = df[column_name1].mean()
    print("")
    RMSE = np.sqrt(MSE)
    MAE = df[column_name2].mean()
    MAPE = df[column_name3].mean()

    # 将结果添加到results_df中
    results_df = results_df._append({
        '特征': features_zh[index],
        'MSE': MSE,
        'RMSE': RMSE,
        'MAE': MAE,
        'MAPE': MAPE
    }, ignore_index=True)

    index += 1

# 将结果写入新的Excel文件
output_path = 'output_file1.xlsx'  # 输出文件的路径
results_df.to_excel(output_path, index=False)  # index=False表示不写入行索引