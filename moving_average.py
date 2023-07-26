
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def convert_to_pandas(data):
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        # 如果数据已经是Pandas DataFrame或Series，则无需转换，直接返回
        return data
    else:
        # 否则，将数据转换为Pandas DataFrame
        if isinstance(data, list):
            # 如果传入的是列表，则将其转换为DataFrame
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # 如果传入的是字典，则将其转换为DataFrame
            return pd.DataFrame.from_dict(data)
        else:
            # 其他情况，将数据作为单列Series转换为DataFrame
            return pd.Series(data)


def moving_average(y_origin, predict_num, w_size):
    y_origin = convert_to_pandas(y_origin)
    # 前 (w_size-1) 个值为 NAN
    y_fitted = y_origin.rolling(window=w_size).mean().fillna(0)

    temp = list(y_origin[-w_size:])
    y_hat = []
    for i in range(predict_num):
        cur_pre = np.mean(temp[-w_size:])
        y_hat.append(cur_pre)
        temp.append(cur_pre)
    return y_fitted, y_hat