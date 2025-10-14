import pandas as pd
import sys
import numpy as np

import os

def extract_columns(input_csv, output_csv, columns_of_interest):
    """
    从CSV文件中提取指定的列，并按指定顺序保存为新的CSV文件
    
    参数:
        input_csv: 输入CSV文件的路径
        output_csv: 输出CSV文件的路径
        columns_of_interest: 关注的列名列表，顺序将作为输出的列顺序
    """
    # 读取CSV文件
    df = pd.read_csv(input_csv)
    
    # 检查关注的列是否都存在于CSV中
    missing_columns = [col for col in columns_of_interest if col not in df.columns]
    if missing_columns:
        print(f"警告: 以下列在CSV文件中不存在，将被忽略: {missing_columns}")
    
    # 过滤出存在的列，并按指定顺序排列
    existing_columns = [col for col in columns_of_interest if col in df.columns]
    if not existing_columns:
        raise ValueError("没有找到任何匹配的列，无法生成输出文件")
    
    # 选择列并保持指定顺序
    result_df = df[existing_columns]

    result_df['D'] = [130000000000] * len(result_df)
    clist = result_df.columns.to_list()
    temp = clist[-1]
    clist[-1] = clist[-2]
    clist[-2] = temp
    result_df = result_df[clist]
    result_df.rename(columns={'total_parameter_count':'N'}, inplace=True)
    result_df.rename(columns={result_df.columns[-1]:'loss'}, inplace=True)
    # 保存为新的CSV文件
    result_df.to_csv(output_csv, index=False)
    print(f"成功生成输出文件: {output_csv}")
    print(f"提取的列: {existing_columns}")

"""
feature = ['step', 'num_experts','d_model','num_blocks','num_heads','kqv_size','flop_increase','k','router_type','seed','routing_frequency','total_parameter_count','dense_parameter_count','flops_per_step']
tars = ['loss_c4','loss_curation_corpus','loss_lambada','loss_pile','loss_validation','loss_wikitext103']
data_path = 'dataset/scaling_for_routing/curves.csv'
df = pd.read_csv(data_path)
output_path_dir = 'dataset/scaling_for_routing/processed_data'
for tar in tars:
    print(f"Processing target: {tar}")
    clms = feature + [tar]
    output_path = os.path.join(output_path_dir, f'curves_{tar}.csv')
    extract_columns(data_path, output_path, clms)"""


import pandas as pd
import os

def split_csv_by_multiple_cols(input_csv, group_columns, output_dir=None):
    """
    按多列的相同取值组合拆分CSV，每个组合单独保存为一个CSV文件
    修复了f-string中包含反斜杠的语法错误
    """
    try:
        df = pd.read_csv(input_csv)
        
        # 检查分组列是否存在
        missing_cols = [col for col in group_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"CSV中缺少以下分组列：{missing_cols}，请检查列名")
        
        # 确定输出目录
        if output_dir is None:
            output_dir = os.path.dirname(input_csv) or '.'
        os.makedirs(output_dir, exist_ok=True)
        
        # 按多列分组
        grouped = df.groupby(group_columns, as_index=False)
        
        # 遍历每个分组并保存
        for _, group_df in grouped:
            # 获取当前分组的各列取值
            group_values = {col: group_df[col].iloc[0] for col in group_columns}
            
            # 生成文件名（修复反斜杠问题：先处理值，再传入f-string）
            filename_parts = []
            for col, val in group_values.items():
                # 先处理特殊字符（将反斜杠和斜杠替换为下划线）
                safe_val = str(val).replace('/', '_').replace('\\', '_')  # 移到f-string外部处理
                filename_parts.append(f"{col}={safe_val}")  # 这里不再包含反斜杠
            
            output_filename = "_".join(filename_parts) + ".csv"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存分组数据
            group_df.to_csv(output_path, index=False)
            print(f"已保存：{output_path}（{len(group_df)}行）")
        
        print(f"\n拆分完成，共生成 {len(grouped)} 个文件，保存至：{output_dir}")
    
    except FileNotFoundError:
        print(f"错误：输入文件 '{input_csv}' 不存在")
    except Exception as e:
        print(f"处理失败：{str(e)}")

def one_hot_encode_csv(csv_path, column_name):
    """
    对CSV文件中指定列进行one-hot编码（结果为1/0）并保存结果
    
    参数:
    csv_path (str): CSV文件的路径
    column_name (str): 需要进行one-hot编码的列名
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 检查指定列是否存在
        if column_name not in df.columns:
            raise ValueError(f"列名 '{column_name}' 不存在于CSV文件中")
        
        # 进行one-hot编码
        df_encoded = pd.get_dummies(df, columns=[column_name], prefix=[column_name])
        
        # 将布尔值转换为整数1和0
        # 找到所有新生成的编码列
        encoded_columns = [col for col in df_encoded.columns if col.startswith(f"{column_name}_")]
        df_encoded[encoded_columns] = df_encoded[encoded_columns].astype(int)
        
        # 获取原文件目录和文件名
        dir_path = os.path.dirname(csv_path)
        file_name = os.path.basename(csv_path)
        name, ext = os.path.splitext(file_name)
        
        # 生成新文件名
        new_file_name = f"{name}_encoded{ext}"
        new_file_path = os.path.join(dir_path, new_file_name)
        
        # 保存编码后的CSV文件
        df_encoded.to_csv(new_file_path, index=False)
        
        print(f"成功生成编码后的文件: {new_file_path}")
        return new_file_path
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{csv_path}'")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return None
    
# 示例用法
if __name__ == "__main__":
    clos = ['dense_parameter_count']
    split_csv_by_multiple_cols('dataset/scaling_for_routing/processed_data/curves_loss_validation_encoded.csv', clos, output_dir='dataset/scaling_for_routing/processed_data/split_by_arch')
    
