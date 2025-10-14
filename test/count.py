import pandas as pd
import os

def count_unique_values(csv_file_path, column_name='N'):
    """
    从CSV文件中提取指定列，并统计该列中不同取值的数量
    
    参数:
        csv_file_path (str): CSV文件的路径
        column_name (str): 要提取的列名，默认为'N'
        
    返回:
        int: 不同取值的数量，如果出现错误则返回-1
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(csv_file_path):
            print(f"错误: 文件 '{csv_file_path}' 不存在")
            return -1
        
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        
        # 检查列是否存在
        if column_name not in df.columns:
            print(f"错误: CSV文件中不存在列名 '{column_name}'")
            return -1
        
        # 提取列并获取唯一值的数量
        unique_values = df[column_name].unique()
        count = len(unique_values)
        
        print(f"列 '{column_name}' 中不同的取值有 {count} 个")
        print("这些取值分别是:", unique_values)
        
        return count
        
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")
        return -1


def count_dn_combinations(csv_file_path):
    """
    从CSV文件中提取D和N列，并计算不同的(D, N)组合数量
    
    参数:
        csv_file_path: CSV文件的路径
        
    返回:
        不同组合的数量
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(csv_file_path):
            print(f"错误: 文件 '{csv_file_path}' 不存在")
            return 0
        
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        
        # 检查D和N列是否存在
        required_columns = ['D', 'N']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"错误: CSV文件中缺少以下列: {', '.join(missing_columns)}")
            return 0
        
        # 提取D和N列
        dn_columns = df[['D', 'N']]
        
        # 去除重复的组合
        unique_combinations = dn_columns.drop_duplicates()
        
        # 返回不同组合的数量
        return len(unique_combinations)
        
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")
        return 0


    

if __name__ == "__main__":
    # 请替换为你的CSV文件路径
    csv_path = "dataset/steplaw/dense_lr_bs_loss.csv" # N: [214663680 268304384 429260800 536872960 1073741824 ] ND combination 17
    # csv_path = "dataset/steplaw/moe_lr_bs_loss.csv" # N: [2150612992 2155174912 2156188672]  ND combination 12
    # csv_path = "farseer/Farseer/ipynb/data/1222_full.csv" 
    """[ 201228288  284207616  397772800  568468736  798470400 1125938688
         1608122368 2273495040 3184435200   99620352  140718592  476931840
         5354047488  239005312 2697992704 1911894528  956354688  336994944
         118460160  676012032  167299200 6369572352 1338278400 4504118400
         3816352512]"""
    
    # 调用函数统计不同取值数量
    # count_unique_values(csv_path)
    # print(count_dn_combinations(csv_file_path=csv_path))
    data = [201228288  ,284207616,  397772800,  568468736,  798470400, 1125938688,
         1608122368, 2273495040, 3184435200,   99620352,  140718592,  476931840,
         5354047488,  239005312, 2697992704, 1911894528,  956354688,  336994944,
         118460160,  676012032 , 167299200, 6369572352, 1338278400, 4504118400,
         3816352512]
    data.sort()
    print(data) # 90m ~6billion