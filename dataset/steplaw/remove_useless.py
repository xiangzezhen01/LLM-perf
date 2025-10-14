import pandas as pd

# 配置参数
input_file = "dataset/steplaw/moe_lr_bs_loss.csv"    # 输入文件路径
output_file = "dataset/steplaw/processed_data/moe_lr_bs_loss.csv"  # 输出文件路径
# 在这里指定要保留的列，按想要的顺序排列
# 不在列表中的列会被自动移除
columns_to_keep = ['h','ffnh','numh','numl','lr','bs','ti','D','N','D/N','loss','smooth loss']  # 示例列名
# seq_len,topk,nume,moeh,sed,h,ffnh,numh,numl,lr,bs,ti,loss,smooth loss,average_iter_time,target_iter,final_iter,exp_name,moe_layer_enum_type,mfa_version,numld,Nattn,N,Na,M,kvcache,D,D/N,total_time,Na/N,moe_name
columns_to_keep = ['topk','nume','moeh','sed','h','ffnh','numh','numl','lr','bs','ti','N','Na','M','kvcache','D','D/N','Na/N', 'loss','smooth loss']  # 示例列名

# 读取CSV文件
df = pd.read_csv(input_file)

# 显示原始列信息
print("原始列:", df.columns.tolist())

# 按指定顺序保留列
try:
    df = df[columns_to_keep]
    # 保存结果
    df.to_csv(output_file, index=False)
    print("处理后列:", df.columns.tolist())
    print(f"处理完成，已保存到 {output_file}")
except KeyError as e:
    print(f"错误：文件中不存在列 {e}")
    print("请检查columns_to_keep中的列名是否正确")
    