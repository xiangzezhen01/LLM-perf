import pandas as pd
import os

def split_csv_by_n(input_csv, output_dir=None):
    """
    按CSV中'N'列的相同取值拆分数据，每个取值单独保存为一个CSV文件
    
    参数:
        input_csv: 输入CSV文件的路径
        output_dir: 输出文件的保存目录（默认与输入文件同目录）
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(input_csv)
        
        # 检查'N'列是否存在
        if 'N' not in df.columns:
            raise ValueError("CSV文件中不存在名为'N'的列，请检查列名是否正确")
        
        # 确定输出目录（默认与输入文件同目录）
        if output_dir is None:
            output_dir = os.path.dirname(input_csv) or '.'  # 若输入路径无目录，使用当前目录
        os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在\

        df = df[['N','D','uD','loss']].copy()
        
        # 按'N'列分组，并遍历每个分组
        for n_value, group_df in df.groupby('N'):
            # 生成输出文件名（避免特殊字符，用下划线替换）
            safe_n_value = str(n_value).replace('/', '_').replace('\\', '_')  # 处理路径分隔符等特殊字符
            output_filename = f"data_N={safe_n_value}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存分组数据（不包含索引）
            group_df.to_csv(output_path, index=False)
            print(f"已保存：{output_path}（包含 {len(group_df)} 行数据）")
        
        print(f"\n拆分完成，共生成 {len(df['N'].unique())} 个文件，保存至：{output_dir}")
    
    except FileNotFoundError:
        print(f"错误：输入文件 '{input_csv}' 不存在")
    except Exception as e:
        print(f"处理失败：{str(e)}")

# 示例用法
if __name__ == "__main__":
    # 输入CSV文件路径（替换为你的文件路径）
    input_csv_path = "dataset/data_constrained/data.csv"
    
    # 输出文件保存目录（可选，默认与输入文件同目录）
    output_directory = "dataset/data_constrained/split_by_N"  # 会自动创建该目录
    
    # 调用函数拆分文件
    split_csv_by_n(input_csv=input_csv_path, output_dir=output_directory)
