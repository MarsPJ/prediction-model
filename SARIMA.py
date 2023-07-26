import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
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


def SARIMA(y_origin,predict_num, p, d, q, P, D, Q, S):
    # p: 自回归（AR）阶数，表示模型中使用多少个滞后值作为自变量。
    # d: 差分阶数，表示将时间序列应用于多少次差分操作以使其稳定。通常情况下，如果你的时间序列不是平稳的，就需要进行差分。
    # q: 移动平均（MA）阶数，表示模型中使用多少个滞后预测误差作为自变量。
    # seasonal_order=(P, D, Q, S): 这是一个包含四个整数的元组，表示季节性部分的模型阶数。它包括以下四个参数：
    #
    # P: 季节性自回归（SAR）阶数，表示季节性部分中使用多少个滞后值作为自变量。
    # D: 季节性差分阶数，表示将季节性时间序列应用于多少次差分操作以使其稳定。
    # Q: 季节性移动平均（SMA）阶数，表示季节性部分中使用多少个滞后预测误差作为自变量。
    # S: 季节性周期性长度，表示数据的季节性周期性。例如，对于月度数据，如果季节性是12（一年的月份数），则 S = 12。
    sarima_model =SARIMAX(y_origin, order=(p, d, q), seasonal_order=(P, D, Q, S))
    sarima_fit = sarima_model.fit()
    y_fitted = sarima_fit.fittedvalues
    forecast = sarima_fit.get_forecast(steps=predict_num)
    y_hat = forecast.predicted_mean
    return y_fitted, y_hat





