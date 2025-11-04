import pandas as pd
import numpy as np
import os

def calculate_statistics(csv_file_path):
    """
    读取CSV文件，计算每列的中位数和四分位距(IQR)

    参数:
        csv_file_path (str): CSV文件的路径

    返回:
        dict: 包含每列的中位数和四分位距的字典，格式为:
             {
                 '列名1': {'median': 中位数, 'iqr': 四分位距},
                 '列名2': {'median': 中位数, 'iqr': 四分位距},
                 ...
             }
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)

        # 存储结果的字典
        results = {}

        # 遍历每一列计算统计量
        for column in df.columns:
            # 获取列数据
            data = df[column]

            # 计算中位数
            median = data.median()

            # 计算四分位距 (IQR = Q3 - Q1)
            q1 = data.quantile(0.25)  # 第一四分位数
            q3 = data.quantile(0.75)  # 第三四分位数
            iqr = q3 - q1

            # 存储结果
            results[column] = {
                'median': median,
                'iqr': iqr
            }

        return results

    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {csv_file_path}")
    except Exception as e:
        raise Exception(f"处理文件时出错: {str(e)}")

if __name__ == '__main__':
    dir_path = 'D:\\CodePycharm\\online_dal-main\\temp\\temp_csv\\test_constrain'
    rank_path = 'D:\\CodePycharm\\online_dal-main\\temp\\result\\constrain'

    for llm_dir in os.listdir(dir_path):
        result = {}
        llm_dir_path = os.path.join(dir_path, llm_dir)
        for file in os.listdir(llm_dir_path):
            file_path = os.path.join(llm_dir_path, file)
            stats = calculate_statistics(file_path)
            for method, stat in stats.items():
                if method not in result.keys():
                    result[method] = []
                median = stat['median']
                iqr = stat['iqr']

                ranks = pd.read_csv(os.path.join(rank_path, file))
                r = ranks.loc[ranks['technique'] == method, 'rank'].values[0]
                result[method].append(f'{r}_{median}_({iqr})')
                print()
        result = pd.DataFrame(result)
        result.to_csv(f'D:\\CodePycharm\\online_dal-main\\temp\\result\\constrain_stats\\{llm_dir}_result.csv', index=False)
