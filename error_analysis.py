import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_mape(actual, predicted):
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.abs((actual - predicted) / actual)
    mape[~np.isfinite(mape)] = np.nan  # 将无效值设为NaN
    return np.nanmean(mape)  # 求均值时忽略NaN值


def getError(y_origin, y_fitted):
    MAE = mean_absolute_error(y_origin, y_fitted)
    MSE = mean_squared_error(y_origin, y_fitted)
    MAPE = calculate_mape(y_origin, y_fitted)
    return MAE, MSE, MAPE