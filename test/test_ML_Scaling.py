import os
import numpy as np
import pandas as pd
from Method.Scaling_method.sacling import farseer, Chinchilla, Kaplan
import logging
import time 
from sklearn.ensemble import RandomForestRegressor

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

set_logging()
finnal_df = pd.DataFrame({'farseer':[], "chinchilla":[], "kaplan":[], "farseer_default":[], "chinchilla_default":[], "kaplan_default":[], "rf_ND":[], "rf_all":[]})
logging.info(f"======================== a new test case ========================")
for name, dir_path in system_dir_dict.items():
    logging.info(f"============ test dataset dir {name} ============")
    for llm_file_name in os.listdir(dir_path):
        single_df = pd.DataFrame({'farseer':[], "chinchilla":[], "kaplan":[], "farseer_default":[], "chinchilla_default":[], "kaplan_default":[], "rf_ND":[], "rf_all":[]})

        logging.info(f"======== test llm name: {llm_file_name} ========")
        llm_file_path = os.path.join(dir_path, llm_file_name)
        test_df = pd.read_csv(llm_file_path)
        all_train_data = []
        for other_train_name in os.listdir(dir_path):
            if other_train_name == llm_file_name:
                continue
            other_llms_path = os.path.join(dir_path, other_train_name)
            df = pd.read_csv(other_llms_path)
            all_train_data.append(df)
        train_df = pd.concat(all_train_data, ignore_index=True)
        logging.info(f"train data shape: {train_df.shape}")
        logging.info(f"test data shape: {test_df.shape}")

        train_np = train_df.values
        test_np = test_df.values
        train_x, train_y = get_attributes_result(train_np)
        test_x, test_y = get_attributes_result(test_np)
        logging.info(f"test data ground truth: {test_y}")

        logging.info(f"==== farseer sacling law ====")
        farseer_model = farseer()
        farseer_model.fit(train_df)
        farseer_result = farseer_model.predict(test_df)
        default_farseer_result = farseer_model.predict(test_df, use_default_params=True)
        farseer_mape = get_mape(test_y, farseer_result)
        farseer_ape_list = get_ape_list(test_y, farseer_result)
        default_farseer_ape_list = get_ape_list(test_y, default_farseer_result)
        default_farseer_mape = get_mape(test_y, default_farseer_result)
        logging.info(f"the farseer pre result: {farseer_result}")
        logging.info(f"the farseer default pre result: {default_farseer_result}")
        logging.info(f"the farseer pre mape: {farseer_mape}")
        logging.info(f"the farseer default pre mape: {default_farseer_mape}")

        logging.info(f"==== Chinchilla ====")
        chinchilla_model = Chinchilla()
        chinchilla_model.fit(train_df)
        chinchilla_result = chinchilla_model.predict(test_df)
        default_chinchilla_result = chinchilla_model.predict(test_df, use_default_params=True)
        default_chinchilla_mape = get_mape(test_y, default_chinchilla_result)
        chinchilla_mape = get_mape(test_y, chinchilla_result)
        chinchilla_ape_list = get_ape_list(test_y, chinchilla_result)
        default_chinchilla_ape_list = get_ape_list(test_y, default_chinchilla_result)
        logging.info(f"the chinchilla pre result: {chinchilla_result}")
        logging.info(f"the chinchilla default pre result: {default_chinchilla_result}")
        logging.info(f"the chinchilla pre mape: {chinchilla_mape}")
        logging.info(f"the chinchilla default pre mape: {default_chinchilla_mape}")

        logging.info(f"==== kaplan ====")
        kaplan_model = Kaplan()
        kaplan_model.fit(train_df)
        kaplan_result = kaplan_model.predict(test_df)
        default_kaplan_result = kaplan_model.predict(test_df, use_default_params=True)
        default_kaplan_mape = get_mape(test_y, default_kaplan_result)
        kaplan_mape = get_mape(test_y, kaplan_result)
        kaplan_ape_list = get_ape_list(test_y, kaplan_result)
        default_kaplan_ape_list = get_ape_list(test_y, default_kaplan_result)
        logging.info(f"the kaplan pre result: {kaplan_result}")
        logging.info(f"the kaplan default pre result: {default_kaplan_result}")
        logging.info(f"the kaplan pre mape: {kaplan_mape}")
        logging.info(f"the kaplan default pre mape: {default_kaplan_mape}")

        logging.info(f"==== random forest ====")
        rf_model = RandomForestRegressor()
        rf_model.fit(train_x, np.ravel(train_y))
        rf_result = rf_model.predict(test_x)
        rf_mape = get_mape(test_y, rf_result)
        rf_ape_list = get_ape_list(test_y, rf_result)
        logging.info(f"the rf pre result: {rf_result}")
        logging.info(f"the rf pre mape: {rf_mape}")

        logging.info(f"==== random forest with only N, D ====")
        rf_ND_model = RandomForestRegressor()
        rf_ND_model.fit(train_df[['N', 'D']], np.ravel(train_y))
        rf_ND_result = rf_ND_model.predict(test_df[['N', 'D']])
        rf_ND_mape = get_mape(test_y, rf_ND_result)
        rf_ND_ape_list = get_ape_list(test_y, rf_ND_result)
        logging.info(f"the rf_ND pre result: {rf_ND_result}")
        logging.info(f"the rf_ND pre mape: {rf_ND_mape}")
        new_row = pd.DataFrame([[farseer_mape, chinchilla_mape, kaplan_mape, default_farseer_mape, default_chinchilla_mape, default_kaplan_mape, rf_ND_mape, rf_mape]], columns=['farseer', 'chinchilla', 'kaplan', 'farseer_default', 'chinchilla_default', 'kaplan_default', 'rf_ND', 'rf_all'])
        finnal_df = pd.concat([finnal_df, new_row], ignore_index=True)
        finnal_df.to_csv('result/25-9-3/result.csv')

        single_df = pd.DataFrame({'farseer':farseer_ape_list, "chinchilla":chinchilla_ape_list, "kaplan":kaplan_ape_list, "farseer_default":default_farseer_ape_list, "chinchilla_default":default_chinchilla_ape_list, "kaplan_default":default_kaplan_ape_list, "rf_ND":rf_ND_ape_list, "rf_all":rf_ape_list}, columns=['farseer', 'chinchilla', 'kaplan', 'farseer_default', 'chinchilla_default', 'kaplan_default', 'rf_ND', 'rf_all'])
        single_df.to_csv(f'result/25-9-3/{name}_{llm_file_name}_result.csv')



