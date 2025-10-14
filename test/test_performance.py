import pandas as pd
import numpy as np
from Method.Scaling_method.sacling import farseer
import logging
import time

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


def test_llms_data(data_path, test_train_rate = 0.1):
    all_data = pd.read_csv(data_path)
    train_data, test_data = split_train_test(all_data, test_train_rate)
    farseer_model = farseer()
    farseer_model.fit(train_data)
    error_list = farseer_model.predict(test_data)
    print(np.mean(error_list))
    logging.info(f"the mape result: {np.mean(error_list)}")

def set_logging():
    logging.basicConfig(filename='test/log.txt',
                        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                        level=logging.INFO, filemode='w')
    
for i in range(30):
    set_logging()
    logging.info(f'============this is the {i} th test============')
    start_time = time.time()
    test_llms_data('farseer/Farseer/ipynb/data/1222_full.csv')
    end_time = time.time()
    logging.info(f"time cost {end_time-start_time} s")

