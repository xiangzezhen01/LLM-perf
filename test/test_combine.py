import os
import numpy as np
import pandas as pd
import sys
from Method.Scaling_method.sacling import Farseer, Chinchilla, Kaplan
import logging
import time 
from sklearn.ensemble import RandomForestRegressor
import random
from sklearn.model_selection import train_test_split

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

test_n_arch = [2,4,8,16]

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
                logging.info(f"train data shape: {train_df.shape}")
                logging.info(f"test data shape: {test_df.shape}")

                all_train_np = train_df.values
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
                farseer_model = Farseer()
                farseer_model.fit(train_df)
                farseer_result = farseer_model.predict(val_df)
                farseer_valid_mape = get_mape(val_y, farseer_result)
                logging.info(f"the farseer pre result: {farseer_result}")


                logging.info(f"==== Chinchilla ====")
                chinchilla_model = Chinchilla()
                chinchilla_model.fit(train_df)
                chinchilla_result = chinchilla_model.predict(val_df)
                chinchilla_valid_mape = get_mape(val_y, chinchilla_result)
                logging.info(f"the chinchilla pre result: {chinchilla_result}")

                logging.info(f"==== Kaplan ====")
                kaplan_model = Kaplan()
                kaplan_model.fit(train_df)
                kaplan_result = kaplan_model.predict(val_df)
                kaplan_valid_mape = get_mape(val_y, kaplan_result)
                logging.info(f"the kaplan pre result: {kaplan_result}")

                logging.info(f"==== random forest with only N, D ====")
                rf_ND_model = RandomForestRegressor()
                rf_ND_model.fit(train_df[['N', 'D']], train_df[train_df.columns[-1]])
                rf_ND_result = rf_ND_model.predict(val_df[['N', 'D']])
                rf_ND_valid_mape = get_mape(val_y, rf_ND_result)
                logging.info(f"the rf_ND pre result: {rf_ND_result}")


                logging.info(f"farseer mape: {farseer_valid_mape}, chinchilla mape: {chinchilla_valid_mape}, kaplan mape: {kaplan_valid_mape}, rf mape: {rf_ND_valid_mape}")

                chinchilla_w = 1-(chinchilla_valid_mape/(farseer_valid_mape + chinchilla_valid_mape + kaplan_valid_mape + rf_ND_valid_mape))
                farseer_w = 1-(farseer_valid_mape/(farseer_valid_mape + chinchilla_valid_mape + kaplan_valid_mape + rf_ND_valid_mape))
                kaplan_w = 1-(kaplan_valid_mape/(farseer_valid_mape + chinchilla_valid_mape + kaplan_valid_mape + rf_ND_valid_mape))
                rf_w = 1-(rf_ND_valid_mape/(farseer_valid_mape + chinchilla_valid_mape + kaplan_valid_mape + rf_ND_valid_mape))

                all_predict = []

                logging.info(f"==== train with all train data ====")
                chinchilla_model = Chinchilla()
                chinchilla_model.fit(all_train_df)
                chinchilla_result = chinchilla_model.predict(all_train_df)
                # chinchilla_mape = get_mape(test_y, chinchilla_result)
                chinchilla_ape_list = get_ape_list(all_train_df[all_train_df.columns[-1]], chinchilla_result)
                kaplan_model = Kaplan()
                kaplan_model.fit(all_train_df)
                kaplan_result = kaplan_model.predict(all_train_df)
                # kaplan_mape = get_mape(test_y, kaplan_result)
                kaplan_ape_list = get_ape_list(all_train_df[all_train_df.columns[-1]], kaplan_result)
                farseer_model = Farseer()
                farseer_model.fit(all_train_df)
                farseer_result = farseer_model.predict(all_train_df)
                # farseer_mape = get_mape(test_y, farseer_result)
                farseer_ape_list = get_ape_list(all_train_df[all_train_df.columns[-1]], farseer_result)

                rf_ND_model = RandomForestRegressor()
                rf_ND_model.fit(all_train_df[['N', 'D']], all_train_df[all_train_df.columns[-1]])
                rf_ND_result = rf_ND_model.predict(all_train_df[['N', 'D']])
                # rf_ND_mape = get_mape(test_y, rf_ND_result)
                rf_ND_ape_list = get_ape_list(all_train_df[all_train_df.columns[-1]], rf_ND_result) 

                combine_result = chinchilla_w * np.array(chinchilla_ape_list) + farseer_w * np.array(farseer_ape_list) + kaplan_w * np.array(kaplan_ape_list) + rf_w * np.array(rf_ND_ape_list)
                combine_loss = (all_train_df[all_train_df.columns[-1]].values - np.array(combine_result))
                all_train_df_loss = all_train_df.copy()
                all_train_df_loss[all_train_df.columns[-1] ] = combine_loss

                rf_futures = RandomForestRegressor()
                rf_futures.fit(all_train_df_loss.drop(all_train_df_loss.columns[-1]), all_train_df_loss[all_train_df_loss.columns[-1]])

                







