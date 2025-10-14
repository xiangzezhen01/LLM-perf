import pandas as pd
import os

def analyze_algorithm_errors(file_path):
    """
    分析算法误差结果CSV文件
    
    参数:
    file_path (str): CSV文件路径
    
    返回:
    dict: 包含中位数和最小损失占比的分析结果
    """
    # 读取CSV文件，第一列作为索引
    df = pd.read_csv(file_path, index_col=0)
    
    # 计算每列的中位数
    medians = df.median()
    
    # 找出每行中损失最小的算法
    min_per_row = df.idxmin(axis=1)
    
    # 统计每个算法成为最小损失的次数
    min_counts = min_per_row.value_counts()
    
    # 计算占比
    total_rows = len(df)
    min_percentages = (min_counts / total_rows) * 100
    
    # 整理结果
    result = {
        'medians': medians.to_dict(),
        'min_percentages': min_percentages.to_dict()
    }
    
    return result

# 使用示例
if __name__ == "__main__":
    result_dir = "result/test_combine"
    for file in os.listdir(result_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(result_dir, file)
            try:
                analysis_result = analyze_algorithm_errors(file_path)
                print(f"文件: {file}")
                print("各算法的中位数误差:")
                for algorithm, median in sorted(analysis_result['medians'].items()):
                    print(f"{algorithm}: {median:.6f}")
                
                print("\n各算法在所有数据中损失最低的占比:")
                for algorithm, percentage in sorted(analysis_result['min_percentages'].items()):
                    print(f"{algorithm}: {percentage:.2f}%")
                print("\n" + "="*50 + "\n")
            except Exception as e:
                print(f"分析文件 {file} 过程中出现错误: {str(e)}")  

