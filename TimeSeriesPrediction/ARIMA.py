import pandas as pd
import statsmodels.api as sm
import numpy as np


def ARIMA(y_origin, predict_num, d, p, q):
    model = sm.tsa.ARIMA(y_origin, order=(d, p, q))
    results = model.fit()
    # # 获取拟合值和预测值
    y_fitted = results.fittedvalues
    # forecast()方法的返回值为一个元组，第一个元素是预测值数组，第二个元素是置信区间数组。
    y_hat = results.forecast(steps=predict_num)[0]
    return y_fitted, y_hat

