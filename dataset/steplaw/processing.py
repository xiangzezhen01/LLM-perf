import pandas as pd
import os

# 读取CSV文件（假设文件名为'dense_lr_bs_loss.csv'）
df = pd.read_csv('dense_lr_bs_loss.csv')
df = pd.read_csv('moe_lr_bs_loss.csv')
# 确认架构定义的列名（根据您的数据描述）
architecture_columns = ['h', 'ffnh', 'numh', 'numl']
# 核心架构参数
architecture_columns = [
    'h',
    'ffnh',
    'numh',
    'numl',
    'moeh',  # 专家网络中的前馈层维度
    'nume',
]
# 创建输出目录
output_dir = "moe_model_architectures"
os.makedirs(output_dir, exist_ok=True)

# 按架构分组并保存每组数据
for (h_value, ffnh_value, numh_value, numl_value,moeh_value,nume_value), group in df.groupby(architecture_columns):
    arch_desc = arch_id = f"h_{h_value}_ffnh_{ffnh_value}_numh_{numh_value}_numl_{numl_value}_moeh_{moeh_value}_nume_{nume_value}"

    filename = f"moe_arch_{arch_desc}.csv"  # 限制文件名长度
    filepath = os.path.join(output_dir, filename)

    # 保存该架构的数据到CSV
    group.to_csv(filename, index=False)

print("\nProcessing completed. All architecture-specific files saved to", output_dir)
