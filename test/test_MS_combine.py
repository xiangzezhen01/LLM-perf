from Method.Scaling_method.sacling import Farseer, Chinchilla, Kaplan, combine_scaling
import os
import numpy as np
import pandas as pd
import sys
import logging
import time 
from sklearn.ensemble import RandomForestRegressor
import random
from sklearn.model_selection import train_test_split

data_path = 'dataset/process'
system_dir_dict = {'dense1_loss': 'dataset/process/dense1_loss_arch',
                   'dense1_smooth': 'dataset/process/dense1_smooth_arch',
                   'dense2_loss': 'dataset/process/dense2_loss_arch', 
                   'dense2_smooth': 'dataset/process/dense2_smooth_arch',
                   'moe_loss': 'dataset/process/moe_loss_arch',
                   'moe_smooth': 'dataset/process/moe_sooth_arch'
                   }

def set_logging():
    logging.basicConfig(filename='test/ML_SCALINGLAW_LOG.txt',
                        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s- %(funcName)s',
                        level=logging.INFO, filemode='w')


def get_whole_data(path):
    df = pd.read_csv(path)
    ndarray = df.values
    return ndarray


def get_attributes_result(data):
    X = data[:, :-1]
    Y = data[:, -1][:, np.newaxis]
    return X, Y

def get_mape(ground_truth, pre_result):
    mape_result = []
    for gt, pre in zip(ground_truth, pre_result):
        if gt == 0:
            continue
        mape = abs(gt - pre)/gt
        mape_result.append(mape)
    return np.mean(mape_result)

def get_ape_list(ground_truth, pre_result):
    mape_result = []
    for gt, pre in zip(ground_truth, pre_result):
        if gt == 0:
            continue
        mape = abs(gt - pre)/gt
        mape_result.append(mape)
    return mape_result

test_n_arch = [8,16]

random_time = 5
set_logging()
for times in range(random_time):
    logging.info(f"======================== random time {times} ========================")
    for n_arch in test_n_arch:
        logging.info(f"======================== a new test case: n_arch = {n_arch} ========================")
        finnal_df = pd.DataFrame({'farseer':[], "chinchilla":[], "kaplan":[], "farseer_default":[], "chinchilla_default":[], "kaplan_default":[], "rf_ND":[], "rf_all":[]})
        logging.info(f"======================== a new test case ========================")

        for name, dir_path in system_dir_dict.items():
            logging.info(f"============ test dataset dir {name} ============")
            for llm_file_name in os.listdir(dir_path):

                logging.info(f"======== test llm name: {llm_file_name} ========")
                llm_file_path = os.path.join(dir_path, llm_file_name)
                test_df = pd.read_csv(llm_file_path)
                all_train_data = []
                """for other_train_name in os.listdir(dir_path):
                    if other_train_name == llm_file_name:
                        continue
                    if len(all_train_data) >= 2:
                        break
                    
                    other_llms_path = os.path.join(dir_path, other_train_name)
                    df = pd.read_csv(other_llms_path)
                    all_train_data.append(df)"""
                file_list = os.listdir(dir_path)
                file_list.remove(llm_file_name)
                file_len = len(file_list)
                if n_arch > file_len:
                    n_arch = file_len
                random_index = random.sample(range(file_len), n_arch)
                for index in random_index:
                    other_train_name = file_list[index]
                    other_llms_path = os.path.join(dir_path, other_train_name)
                    df = pd.read_csv(other_llms_path)
                    all_train_data.append(df)

                all_train_df = pd.concat(all_train_data, ignore_index=True)

                all_train_np = all_train_df.values
                all_test_np = test_df.values

                all_train_x, all_train_y = get_attributes_result(all_train_np)
                test_x, test_y = get_attributes_result(all_test_np)

                train_df, val_df = train_test_split(
                all_train_df, 
                test_size=0.1,  # 验证集比例
                train_size=0.9, # 训练集比例
                random_state=42, # 随机种子，保证结果可重现
                shuffle=True    # 是否在划分前打乱数据，默认为True
                )
                val_y = val_df.values[:, -1]

                cs_model = combine_scaling(['farseer', 'chinchilla', 'kaplan', 'rf_nd'])
                cs_model.fit(train_df)
                cs_result = cs_model.predict(val_df)
                cs_mape = get_mape(val_y, cs_result)
                cs_ape_list = get_ape_list(val_y, cs_result)
                print(f"combine scaling law mape is {cs_mape}")