import os
import pandas as pd

def process_csv_files(directory_path):
    """
    处理指定目录下的所有CSV文件，过滤掉'loss'列为空的行，并保存为新文件
    
    参数:
        directory_path (str): 包含CSV文件的目录路径
    """
    # 检查目录是否存在
    if not os.path.isdir(directory_path):
        print(f"错误: 目录 '{directory_path}' 不存在")
        return
    
    # 获取目录下所有CSV文件
    csv_files = [f for f in os.listdir(directory_path) 
                if f.endswith('.csv') and os.path.isfile(os.path.join(directory_path, f))]
    
    if not csv_files:
        print(f"在目录 '{directory_path}' 中未找到CSV文件")
        return
    
    # 处理每个CSV文件
    for file in csv_files:
        file_path = os.path.join(directory_path, file)
        
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 检查是否存在'loss'列
            if 'loss' not in df.columns:
                print(f"警告: 文件 '{file}' 中不包含 'loss' 列，已跳过")
                continue
            
            # 过滤掉'loss'列为空的行
            # 这里使用pd.notna()来判断非空值，包括排除NaN、None等
            filtered_df = df[pd.notna(df['loss'])]
            
            # 生成新文件名，在原文件名前添加'filtered_'
            file_name, file_ext = os.path.splitext(file)
            new_file_name = f"filtered_{file_name}{file_ext}"
            new_file_path = os.path.join(directory_path, new_file_name)
            
            # 保存过滤后的DataFrame为新CSV文件
            filtered_df.to_csv(new_file_path, index=False)
            
            # 输出处理信息
            removed_rows = len(df) - len(filtered_df)
            print(f"已处理文件: {file}")
            print(f"  原始行数: {len(df)}")
            print(f"  过滤后行数: {len(filtered_df)}")
            print(f"  移除空loss行: {removed_rows}")
            print(f"  保存为: {new_file_name}\n")
            
        except Exception as e:
            print(f"处理文件 '{file}' 时出错: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 替换为你的CSV文件所在目录路径
    target_directory = "dataset/process/moe2_routing"
    process_csv_files(target_directory)
