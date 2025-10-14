import numpy as np
from lmfit import Model, Parameters
import pandas as pd

# 定义函数模型
"""def scaling_law(N, D, a1, alpha, b1, a2, beta, b2, a3, gamma, b3):
    exp_term1 = np.exp(a1 * N**alpha + b1)
    exp_term2 = np.exp(a2 * N**beta + b2)
    exp_term3 = np.exp(a3 * N**gamma + b3)
    return exp_term3 + exp_term2 * (D ** (-np.exp(exp_term1)))"""
def scaling_law(N, D, a1, alpha, b1, a2, beta, b2, a3, gamma, b3):
    # 计算第一组指数内的线性组合（a1*N^α + b1）
    inner_exp1 = a1 * N**alpha + b1
    
    # 计算第二组指数（a2*N^β + b2）
    inner_exp2 = a2 * N**beta + b2
    
    # 计算第三组指数（a3*N^γ + b3）
    inner_exp3 = a3 * N**gamma + b3
    
    # 计算各项的指数
    exp_term1 = np.exp(inner_exp1)  # e^(a1*N^α + b1)
    exp_term2 = np.exp(inner_exp2)  # e^(a2*N^β + b2)
    exp_term3 = np.exp(inner_exp3)  # e^(a3*N^γ + b3)
    
    # 计算D的项：D^(-e^(a1*N^α + b1))
    D_power = D ** (-exp_term1)
    
    return exp_term3 + exp_term2 * D_power

def load_data(df):
    clomns_filtered_data = df[['N', 'D', 'loss']]    
    return clomns_filtered_data

def split_train_test(df, test_ratio = 0.2):
    """
    将DataFrame按比例随机划分为训练集和测试集
    """
    if not (0 < test_ratio <= 1):
        raise ValueError("测试集比例必须在(0, 1)之间")
    
    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    

    return df.iloc[train_indices], df.iloc[test_indices]

# 创建模型
model = Model(scaling_law, independent_vars=['N','D'])

# 设置参数初始值和约束
params = Parameters()
params.add('a1', value= -0.5, min=-1, max=0)
params.add('alpha', value=0.5, min=0, max=1)
params.add('b1', value=0.5, min=0, max=1)
params.add('a2', value=50, min=0, max=100)
params.add('beta', value= -0.5, min = -1, max = 0)
params.add('b2', value=-5, min=-10, max = 0)
params.add('a3', value=-0.5, min=-1, max=0)
params.add('gamma', value=0.5, min=0, max=1)
params.add('b3', value=-0.5, min = -1, max=0)

# 加载数据
path = 'farseer/Farseer/ipynb/data/1222_full.csv'
df = pd.read_csv(path)
train_data, test_data = split_train_test(df)
L_data = train_data['loss']
N_data = train_data['N']
D_data = train_data['D']

# 加载实验数据 (N_data, D_data, L_data)
result = model.fit(L_data, params, method='differential_evolution', N=N_data, D=D_data)

# 输出结果
print(result.fit_report())
# result.plot()  # 可视化拟合效果
print()