import os
import numpy as np
import pandas as pd
import sys
from Method.Scaling_method.sacling import Farseer, Chinchilla, Kaplan, combine_scaling
import logging
import time 
from sklearn.ensemble import RandomForestRegressor
import random
# 

# 'dense1_loss': 'dataset/process/dense1_loss_arch', 
# 'dense3_loss': 'dataset/process/data_constain_dense', 
# 'dense2_loss': 'dataset/process/dense2_loss_arch',    
# 'moe_loss': 'dataset/process/moe_loss_arch',  
# 'moe2_loss': 'dataset/process/moe2_routing',   
system_dir_dict = {
    
                    'dense1_loss': 'dataset/process/dense1_loss_arch', 
                    'dense3_loss': 'dataset/process/data_constain_dense', 
                    'dense2_loss': 'dataset/process/dense2_loss_arch',    
                    'moe_loss': 'dataset/process/moe_loss_arch',  
                    'moe2_loss': 'dataset/process/moe2_routing', 
                   'dense4_loss': 'dataset/process/data_mixlaw_dense23',
                   'dense5_loss': 'dataset/process/data_mixlaw_dense5'
                   }
# 

system_test_arch_dict = {
                        'dense4_loss': [1,2,4],
                        'dense5_loss': [1,2,4],
                        'dense1_loss': [1,5,12,18,24],
                        'dense3_loss': [1,3,9,15,18],
                        'dense2_loss': [1,2,3,4], 
                        'moe_loss': [1,2,3],
                        'moe2_loss': [1,2,3,5]
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



def test_methods(dir_path, n_arch):
    result_dict = {'farseer':[], "chinchilla":[], "kaplan":[], "rf_ND":[], "rf_all":[], 'combine_scaling_softmax':[], 'combine_scaling_softmax_constrain':[]}
    for llm_file_name in os.listdir(dir_path):

        logging.info(f"======== test llm name: {llm_file_name} ========")
        print(f"test llm name: {llm_file_name}")
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
        select_n = n_arch
        if n_arch > file_len:
            select_n = file_len
        random_index = random.sample(range(file_len), select_n)
        for index in random_index:
            other_train_name = file_list[index]
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
        farseer_model = Farseer()
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

        logging.info(f"==== combine scaling law with softmax ====")
        cs_model = combine_scaling(['farseer', 'chinchilla', 'kaplan', 'rf_nd'], data_constarin = False)
        cs_model.fit(train_df)
        cs_result = cs_model.predict(test_df)
        cs_mape = get_mape(test_y, cs_result)
        cs_ape_list = get_ape_list(test_y, cs_result)
        logging.info(f"the combine scaling law pre result: {cs_result}")
        logging.info(f"the combine scaling law pre mape: {cs_mape}")

        logging.info(f"==== combine scaling law with data constrain ====")
        cs1_model = combine_scaling(['farseer', 'chinchilla', 'kaplan', 'rf_nd'], data_constarin = True)
        cs1_model.fit(train_df)
        cs1_result = cs1_model.predict(test_df)
        cs1_mape = get_mape(test_y, cs1_result)
        cs1_ape_list = get_ape_list(test_y, cs1_result)
        logging.info(f"the combine scaling law pre result: {cs1_result}")
        logging.info(f"the combine scaling law pre mape: {cs1_mape}")

        result_dict['farseer'].append(farseer_mape)
        result_dict['chinchilla'].append(chinchilla_mape)
        result_dict['kaplan'].append(kaplan_mape)
        result_dict['rf_ND'].append(rf_ND_mape)
        result_dict['rf_all'].append(rf_mape)
        result_dict['combine_scaling_softmax'].append(cs_mape)
        result_dict['combine_scaling_softmax_constrain'].append(cs1_mape)

    return [result_dict['farseer'], result_dict['chinchilla'], result_dict['kaplan'], result_dict['rf_ND'], result_dict['rf_all'], result_dict['combine_scaling_softmax'], result_dict['combine_scaling_softmax_constrain']]
    

if __name__ == "__main__":
    random.seed(43)
    np.random.seed(43)
    repeat_time = 3
    set_logging()
    for i in range(repeat_time):
        for name, dir_path in system_dir_dict.items():
            logging.info(f"============ test dataset dir {name} ============")
            test_n_arch = system_test_arch_dict[name]
            for n_arch in test_n_arch:
                print(f"start test dataset dir {name}, n_arch = {n_arch}")
                result_dict = {'farseer':[], "chinchilla":[], "kaplan":[], "rf_ND":[], "rf_all":[],'combine_scaling_softmax':[], 'combine_scaling_softmax_constrain':[]}
                logging.info(f"======================== a new test case: n_arch = {n_arch} ========================")
                start_time = time.time()
                result_list = test_methods(dir_path, n_arch)
                end_time = time.time()
                logging.info(f"this test case use time: {end_time - start_time}")
                logging.info(f"the result list is {result_list}")

                result_dict['farseer']=result_list[0]
                result_dict['chinchilla']=result_list[1]
                result_dict['kaplan']=result_list[2]
                result_dict['rf_ND']=result_list[3]
                result_dict['rf_all']=result_list[4]
                result_dict['combine_scaling_softmax']=result_list[5]
                result_dict['combine_scaling_softmax_constrain']=result_list[6]
                
                df = pd.DataFrame(result_dict)
                df.to_csv(f'result/test_constrain_rebuild/{name}_arch_{n_arch}_time{i}.csv', index=False)
                logging.info(f"======================== a new test case end ========================")
                print(f"end test dataset dir {name}, n_arch = {n_arch}")
            logging.info(f"============ test dataset dir {name} end ============")
    print("done")



