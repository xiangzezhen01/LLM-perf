from abc import ABC, abstractmethod
from farseer.Farseer.ipynb.scaling_law_end2end_fit import scaling_fit_fn_an_bn
from farseer.Farseer.scalinglaw_utils.scaling_law_fiting.data_filters import calculate_differences, filter_data_further, filter_data
import math
import pandas as pd
import numpy as np
from lmfit import Model, Parameters
from sklearn.ensemble import RandomForestRegressor
from multiprocessing import Pool
import functools

all_scaling_law = ['kaplan', 'chinchilla', 'farseer', 'rf_nd']

def split_train_test(df, test_ratio = 0.5):
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

class ScalingMethod(ABC):
    """
    Abstract base class for scaling methods.
    """
    @abstractmethod
    def fit(self, **kwargs):
        pass

    def predict(self, **kwargs):
        """
        Predicts the output based on the input variables and fitted parameters.
        """
        pass

class Chinchilla(ScalingMethod):
    def __init__(self):
        self.name = "Chinchilla"
        self.target_name = None
        self.fitted_parameters = {'A': None, 'B': None, 'E': None, 'alpha': None, 'beta': None}
        self.fit_success = False
        

    def fit(self, df):
        
        self.target_name = df.columns.to_list()[-1]
        def scaling_law(N, D, A, B, E, alpha, beta):
            part1 = A/(N**alpha)
            part2 = B/(D**beta)
            part3 = E
            return part1 + part2 + part3
        
        model = Model(scaling_law, independent_vars=['N','D'])

        # 设置参数初始值和约束
        params = Parameters()
        params.add('A', value= 406.4, min=400, max = 410)
        params.add('B', value=410.7, min=405, max = 415)
        params.add('E', value = 1.69, min = 1.2, max = 1.8)
        params.add('alpha', value = 0.34, min = 0.2, max = 0.5)
        params.add('beta', value= 0.28, min = 0.1, max = 0.5)

        L_data = df[self.target_name]
        N_data = df['N']
        D_data = df['D']

        # 加载实验数据 (N_data, D_data, L_data)
        try:
            result = model.fit(L_data, params, method='differential_evolution', N=N_data, D=D_data)
            self.fit_success = True
            self.fitted_parameters = result.best_values
        except Exception as e:  # 这里应该用 except 而不是 catch
            print(f"Chinchilla fitting error: {str(e)}")
            self.fit_success = False
        # 输出结果
        # print(result.fit_report())
        

    def predict(self, df, use_default_params = False):
        pre_result = []
        if use_default_params or (not self.fit_success):
            for index, row in df.iterrows():
                N = row['N']
                D = row['D']
                loss_pre = 406.4/(N**0.34) + 410.7/(D**0.28) + 1.69
                pre_result.append(loss_pre)
        else:    
            for index, row in df.iterrows():
                N = row['N']
                D = row['D']
                loss_pre = self.fitted_parameters['E'] + self.fitted_parameters['A']/(N**self.fitted_parameters['alpha']) + self.fitted_parameters['B']/(D**self.fitted_parameters['beta'])
                pre_result.append(loss_pre)
        return pre_result


class Farseer(ScalingMethod):
    def __init__(self, fit_variant = 'all_trans'):
        self.name = "Farseer"
        self.input_variables = ['N', 'D']
        self.output_variables = ['loss']
        self.fitted_parameters = {'alpha': None, 'beta': None, 'gamma': None, 'a1': None, 'a2': None, 'a3': None, 'b1': None, 'b2': None, 'b3': None}
        self.min_len = 5
        self.fit_variant = fit_variant
        self.target_name = None
        self.fit_success = False


    def fit(self, df):
        self.target_name = df.columns.tolist()[-1]
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
        
        model = Model(scaling_law, independent_vars=['N','D'])

        # 设置参数初始值和约束
        params = Parameters()
        params.add('a1', value= -0.124, min=-1, max=0)
        params.add('alpha', value=0.123, min=0, max=1)
        params.add('b1', value=0.424, min=0, max=1)
        params.add('a2', value=88.01, min=50, max=100)
        params.add('beta', value= -0.1, min = -0.5, max = 0)
        params.add('b2', value=-6.287, min=-10, max = -5)
        params.add('a3', value=-0.021, min=-0.1, max=0)
        params.add('gamma', value=0.169, min=0, max=0.5)
        params.add('b3', value=-0.091, min = -0.5, max=0)

        L_data = df[self.target_name]
        N_data = df['N']
        D_data = df['D']

        # 加载实验数据 (N_data, D_data, L_data)
        try:

            result = model.fit(L_data, params, method='differential_evolution', N=N_data, D=D_data)
            self.fit_success = True
            self.fitted_parameters = result.best_values
        except Exception as e:  # 这里应该用 except 而不是 catch
            print(f"Farseer fitting error: {str(e)}")
            self.fit_success = False
        # 输出结果
        # print(result.fit_report())
        
    """    def fit(self, df, config_result = None):
        if config_result == None:
            df_trans = filter_data(df, min_D=1)
            df_efn, df_result=scaling_fit_fn_an_bn(df_trans, self.min_len, filter_query = "2e8<N<4e9", use_pred=False, label='4e9', enable_constrain=True, fit_variant=self.fit_variant)
        else:
            df_result = pd.read_csv(config_result)
        self.fitted_parameters['alpha'] = df_result['a'][0]
        self.fitted_parameters['beta'] = df_result['b'][0]
        self.fitted_parameters['gamma'] = df_result['q'][0]
        self.fitted_parameters['a1'] = df_result['A'][0]
        self.fitted_parameters['a2'] = df_result['B'][0]
        self.fitted_parameters['a3'] = df_result['s'][0]
        self.fitted_parameters['b1'] = df_result['E'][0]
        self.fitted_parameters['b2'] = df_result['Q'][0]
        self.fitted_parameters['b3'] = df_result['S'][0]"""
    
    def predict(self, df, use_default_params = False):
        """
        Predicts the output based on the input variables and fitted parameters.
        """
        # Implement prediction logic here
        # if self.fit_variant == "all_trans":
        #     return math.e**(self.fitted_parameters['a3']*(N**self.fitted_parameters['gamma'])+self.fitted_parameters['b3']) + math.e**(self.fitted_parameters['a2']*(N**self.fitted_parameters['beta'])+self.fitted_parameters['b2']) * D**(-math.e**(-self.fitted_parameters['a1']*N**(self.fitted_parameters['alpha'])+self.fitted_parameters['b1']))
        pre_result = []
        if use_default_params or (not self.fit_success):
            for index, row in df.iterrows():
                N = row['N']
                D = row['D']
                loss_pre = math.e**(-0.021*(N**0.169)-0.091) + math.e**(88.01*(N**-0.1)-6.287) * D**(-math.e**(-0.124*N**(0.123)+0.424))
                pre_result.append(loss_pre)
        else:
            for index, row in df.iterrows():
                N = row['N']
                D = row['D']
                loss_pre = math.e**(self.fitted_parameters['a3']*(N**self.fitted_parameters['gamma'])+self.fitted_parameters['b3']) + math.e**(self.fitted_parameters['a2']*(N**self.fitted_parameters['beta'])+self.fitted_parameters['b2']) * D**(-math.e**(self.fitted_parameters['a1']*N**(self.fitted_parameters['alpha'])+self.fitted_parameters['b1']))
                pre_result.append(loss_pre)
        return pre_result
    
class Kaplan(ScalingMethod):
    def __init__(self):
        self.target_name = None
        self.name = 'Kaplan'
        self.fitted_parameters = {'aN': None, 'aD': None, 'Nc': None, 'Dc':None}
        self.fit_success = False
    
    def fit(self, df):
        self.target_name = df.columns.to_list()[-1]
        def scaling_law(N, D, aN, aD, Nc, Dc):
            part1 = ((Nc*(10**13))/(N))**(aN/aD)
            part2 = ((Dc*(10**13))/(D))
            return (part1 + part2)**aD
        
        model = Model(scaling_law, independent_vars=['N','D'])

        # 设置参数初始值和约束
        params = Parameters()
        params.add('Nc', value = 6.4, min=0.01, max = 10)
        params.add('Dc', value = 1.8, min=0.01, max=10)
        params.add('aN', value=0.076, min=0.01, max=0.1)
        params.add('aD', value=0.103, min=0.01, max=0.2)

        L_data = df[self.target_name]
        N_data = df['N']
        D_data = df['D']

        # 加载实验数据 (N_data, D_data, L_data)
        try:
            result = model.fit(L_data, params, method='differential_evolution', N=N_data, D=D_data)
            self.fit_success = True
            self.fitted_parameters = result.best_values
        except Exception as e:  # 这里应该用 except 而不是 catch
            print(f"Kaplan fitting error: {str(e)}")
            self.fit_success = False

        # 输出结果
        # print(result.fit_report())
        
    
    def predict(self, df, use_default_params = False):
        pre_result = []
        if use_default_params or (not self.fit_success):
            for index, row in df.iterrows():
                N = row['N']
                D = row['D']
                loss_pre = (((6.4*(10**13))/float(N))**(0.076/0.103) + ((1.8*(10**13))/float(D)))**0.103
                pre_result.append(loss_pre)
        else:
            for index, row in df.iterrows():
                N = row['N']
                D = row['D']
                loss_pre = (((self.fitted_parameters['Nc']*(10**13))/float(N))**(self.fitted_parameters['aN']/self.fitted_parameters['aD']) + ((self.fitted_parameters['Dc']*(10**13))/float(D)))**self.fitted_parameters['aD']
                pre_result.append(loss_pre)
        return pre_result

class ResidualsModel(ScalingMethod):
    def __init__(self, model_name = 'RandomForest'):
        self.name = model_name
        self.model = None
    
    def fit(self, df):
        if self.name == 'RandomForest':
            self.model = RandomForestRegressor()
            residuals_x = df.drop(df.columns[-1], axis=1)
            residuals_y = df[df.columns[-1]]
            self.model.fit(residuals_x, residuals_y)
        else:
            raise ValueError(f"Unknown model name: {self.name}")
    
    def predict(self, df):
        return self.model.predict(df.drop(df.columns[-1], axis=1))

class RF_ND(ScalingMethod):
    def __init__(self):
        self.name = "RandomForest_ND"
        self.model = None
    
    def fit(self, df):
        self.model = RandomForestRegressor()
        self.model.fit(df[['N', 'D']], df[df.columns[-1]])
    
    def predict(self, df):
        return self.model.predict(df[['N', 'D']])
    
def build_scaling_law(key):
    if key == 'kaplan':
        return Kaplan()
    if key == 'chinchilla':
        return Chinchilla()
    if key == 'farseer':
        return Farseer()
    if key == 'rf_nd':
        return RF_ND()
    return None

def softmax_weighting(errors, temperature=1.0):
    """
    根据模型在验证集上的误差计算权重使用Softmax方法
    
    参数:
    errors -- 列表 包含每个模型在验证集上的平均误差
    temperature -- 温度参数
    
    返回:
    weights -- 列表 与输入errors对应的权重列表 权重之和为1
    """
    # 将误差转换为性能分数（误差越小，分数越高）
    scores = -np.array(errors)
    
    # 数值稳定性处理：减去最大值避免指数运算溢出
    max_score = np.max(scores)
    shifted_scores = scores - max_score
    
    # 计算指数项
    exp_scores = np.exp(shifted_scores / temperature)
    
    # 计算Softmax权重（自动归一化）
    weights = exp_scores / np.sum(exp_scores)
    
    return weights.tolist()  


def inverse_error_weighting(errors, epsilon=1e-6):
    """
    根据模型在验证集上的误差计算权重（使用倒数法）
    
    参数:
    errors -- 列表，包含每个模型在验证集上的平均误差
    epsilon -- 防止除零错误的小常数（默认1e-6）
    
    返回:
    weights -- 列表，与输入errors对应的权重列表，权重之和为1
    """
    # 将输入转换为NumPy数组
    errors = np.array(errors)
    
    # 检查是否有负误差值
    if np.any(errors < 0):
        raise ValueError("误差值不能为负数")
    
    # 计算原始权重：1/(error + epsilon)
    raw_weights = 1 / (errors + epsilon)
    
    # 归一化权重（确保权重和为1）
    total_weight = np.sum(raw_weights)
    weights = raw_weights / total_weight
    
    return weights.tolist()

def best_model_weighting(errors):
    """
    选择验证集上表现最好的模型（误差最小的模型），将其权重设为1，其他模型权重设为0
    
    参数:
    errors -- 列表，包含每个模型在验证集上的平均误差
    
    返回:
    weights -- 列表，与输入errors对应的权重列表，其中最佳模型权重为1，其余为0
    """
    # 检查输入是否为空
    if not errors:
        return []
    
    # 找到最小误差的索引（即最佳模型）
    min_error = min(errors)
    best_index = errors.index(min_error)
    
    # 创建全零权重列表
    weights = [0.0] * len(errors)
    
    # 将最佳模型的权重设为1
    weights[best_index] = 1.0
    
    return weights

def proportional_to_best_weighting(errors, epsilon=1e-6):
    """
    根据模型在验证集上的误差计算权重（最小误差比例法）
    
    参数:
    errors -- 列表，包含每个模型在验证集上的平均误差
    epsilon -- 防止除零错误的小常数（默认1e-6）
    
    返回:
    weights -- 列表，与输入errors对应的权重列表，权重之和为1
    """
    # 检查输入是否为空
    if not errors:
        return []
    
    # 检查是否有负误差值
    if any(e < 0 for e in errors):
        raise ValueError("误差值不能为负数")
    
    # 找到最小误差
    min_error = min(errors)
    
    # 处理所有模型误差均为0的情况
    if min_error == 0 and all(e == 0 for e in errors):
        # 所有模型误差相同且为0，则平均分配权重
        return [1.0 / len(errors)] * len(errors)
    
    # 计算原始权重：min_error / (error + epsilon)
    raw_weights = []
    for e in errors:
        # 防止除零错误
        denominator = e + epsilon
        # 计算权重比例
        if min_error == 0:
            # 如果最小误差为0（但非所有模型误差为0），则只有误差为0的模型有权重
            raw_weights.append(1.0 if e == 0 else 0.0)
        else:
            raw_weights.append(min_error / denominator)
    
    # 归一化权重（确保权重和为1）
    total_weight = sum(raw_weights)
    
    # 处理所有权重为0的情况（理论上不应该发生）
    if total_weight == 0:
        return [1.0 / len(errors)] * len(errors)
    
    weights = [w / total_weight for w in raw_weights]
    
    return weights

def split_df_by_order(df, k):
    """
    将DataFrame按顺序平均分为k个DataFrame
    
    参数:
        df: 要分割的原DataFrame
        k: 分割的数量
        
    返回:
        包含k个DataFrame的列表
    """
    if k <= 0:
        raise ValueError("k必须是正整数")
    
    n = len(df)
    if k > n:
        # 当k大于行数时，每个子DataFrame最多1行（可能有空白）
        return [df.iloc[i:i+1] for i in range(k)]
    
    # 计算每个子DataFrame的行数
    base_size = n // k  # 基础行数
    remainder = n % k   # 剩余行数（前remainder个多分1行）
    
    split_dfs = []
    start = 0
    for i in range(k):
        # 前remainder个子DataFrame多1行
        size = base_size + 1 if i < remainder else base_size
        end = start + size
        # 按顺序切片
        split_df = df.iloc[start:end].copy()
        split_dfs.append(split_df)
        start = end
    
    return split_dfs

def evaluate_model(train_df, test_df, model, constrain = False):
    if constrain:
        train_df = scaling_data_constrain(train_df)
    model.fit(train_df)
    predictions = model.predict(test_df)
    true_values = test_df[test_df.columns[-1]].values
    mape = np.mean(np.abs((true_values - predictions) / true_values))
    return mape

def cross_val_score(model, df, k=10, constrain = False):
    # print(f"Starting {k}-fold cross-validation...")
    fold_size = len(df) // k
    mape_result = []
    df = df.sample(frac=1)  # Shuffle the DataFrame
    k_fold_dfs = split_df_by_order(df, k)
    for fold in k_fold_dfs:
        #print(fold)
        mape = evaluate_model(df.drop(fold.index), fold, model, constrain=constrain)
        mape_result.append(mape)
    return np.mean(mape_result)

def _cross_val_wrapper(args):
    model, df, k, c = args
    return cross_val_score(model, df, k, c)

def scaling_data_constrain(df):
    required_columns = ['N', 'D', 'loss']
    if not set(required_columns).issubset(df.columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"输入的DataFrame缺少必要的列: {missing}")
    
    # 按N和D分组，计算每组loss的最小值
    result_df = df.groupby(['N', 'D'])['loss'].min().reset_index()
    
    return result_df

    

class combine_scaling(ScalingMethod):
    def __init__(self, scaling_laws, weight_method = 'softmax', data_constarin = True):
        self.data_constarin = data_constarin
        self.weight_method = weight_method
        self.residuals_model = None
        self.base_scaling_law = []
        self.weights = []
        for sl in scaling_laws:
            model = build_scaling_law(sl)
            if model is not None:
                self.base_scaling_law.append(model)

    def compute_weights(self, error_list):
        if self.weight_method == 'softmax':
            weights = softmax_weighting(error_list)
        elif self.weight_method == 'inverse':
            weights = inverse_error_weighting(error_list)
        elif self.weight_method == 'best':
            weights = best_model_weighting(error_list)
        elif self.weight_method == 'p2best':
            weights = proportional_to_best_weighting(error_list)
        elif self.weight_method == 'equal':
            weights = [1/len(error_list)] * len(error_list)
        else:
            raise ValueError(f"Unknown weight method: {self.weight_method}")
        return weights
    
    def predict_by_weighed_scalings(self, df):
        all_result = []
        for i, model in enumerate(self.base_scaling_law):
            r = model.predict(df)
            wr = np.array(r) * self.weights[i]
            all_result.append(wr)

        weighted_result = np.sum(np.array(all_result), axis=0)
        return weighted_result
    
    def fit(self, train_df):
        if self.data_constarin:
            train_df_scaling = scaling_data_constrain(train_df)
        else:
            train_df_scaling = train_df
        params = [(model, train_df_scaling, 5, self.data_constarin) for model in self.base_scaling_law]
        # 10-fold cross-validation to get each scaling law's validation error
        all_sls_valid_merrors = []
     
        with Pool(processes=4) as pool:
            # 映射参数到交叉验证函数
            all_sls_valid_merrors = pool.map(_cross_val_wrapper, params)

        pool.close()
        pool.join()
        
        # 获取权重
        self.weights = self.compute_weights(all_sls_valid_merrors)
        
        residuals_list = [] 
        # get config and its residuals
        for model in self.base_scaling_law:
            model.fit(train_df_scaling)
        
        scaling_combine_result = self.predict_by_weighed_scalings(train_df)
        residuals_list = train_df[train_df.columns[-1]].values - np.array(scaling_combine_result)
        
        residuals_df = train_df.copy()
        residuals_df.rename(columns={residuals_df.columns[-1]:'residuals'}, inplace=True)
        residuals_df[residuals_df.columns[-1]] = residuals_list
        
        self.residuals_model = ResidualsModel('RandomForest')
        self.residuals_model.fit(residuals_df)
        
    def predict(self, df):
        scaling_combine_result = self.predict_by_weighed_scalings(df)
        residuals_result = self.residuals_model.predict(df)
        final_result = scaling_combine_result + residuals_result
        return final_result
        

# 使用示例
if __name__ == "__main__":
    # 创建示例DataFrame
    data = {
        'N': [1, 1, 2, 2, 2 ,2,2, 3],
        'D': [10, 10, 20, 20, 20,20,30, 30],
        'loss': [5.2, 3.8, 7.1, 6.5,100 ,1, 4.9, 8.3]
    }
    df = pd.DataFrame(data)
    
    # 调用函数
    result = scaling_data_constrain(df)
    print("原始DataFrame:")
    print(df)
    print("\n分组后取loss最小值的结果:")
    print(result)
    
    
    


            
        