from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import numpy as np

"""
trend：指定趋势项的类型，可以取值为 'add' 或 'mul'。'add' 表示加法模型（即线性趋势），'mul' 表示乘法模型（即指数趋势）。

seasonal：指定季节性项的类型，可以取值为 'add' 或 'mul'。'add' 表示加法季节性，'mul' 表示乘法季节性。

seasonal_periods：指定季节性的周期，例如，如果数据每年有四个季度，就设置为4。

在构建 ExponentialSmoothing 对象时，模型会根据传入的 trend 和 seasonal 参数以及指定的 seasonal_periods 值，
自动选择相应的平滑系数来拟合趋势和季节性。具体的平滑系数为 alpha、beta 和 gamma，
它们分别对应 smoothing_level（平滑水平项）、smoothing_trend（平滑趋势项）和 smoothing_seasonal（平滑季节性项）。
"""

def exponential_smoothing(y_origin, predict_num, seasonal=None, trend=None, seasonal_periods=None):
    model = ExponentialSmoothing(y_origin, seasonal=seasonal, trend=trend, seasonal_periods=seasonal_periods)
    result = model.fit()
    y_hat = result.forecast(predict_num)
    y_fitted = result.fittedvalues
    alpha = result.params['smoothing_level']
    beta = result.params['smoothing_trend']
    gamma = result.params['smoothing_seasonal']
    # print(result.params)
    return y_fitted, y_hat, alpha, beta, gamma




