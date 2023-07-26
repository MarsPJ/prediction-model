import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ARIMA
import SARIMA
import error_analysis
import exponential_smoothing
import moving_average

# 读取CSV数据
data = pd.read_excel('data.xlsx')

# 使用条件筛选得到满足条件的行
condition = (data['场地1'] == 14) & (data['场地2'] == 10)
filtered_data = data[condition]

# 将"时间"列转换为DatetimeIndex并设置为索引
filtered_data.index = pd.date_range(start=filtered_data['时间'].min(), periods=len(filtered_data), freq='D')
filtered_data.drop('时间', axis=1, inplace=True)

# 打印筛选后的数据
# print(filtered_data)
y_origin = filtered_data["货运量"]
# 预测数目
predict_num = 31
# ARIMA 模型
# y_fitted, y_hat = ARIMA.ARIMA(y_origin, predict_num, 2, 0, 0)
# 移动平均模型（默认将Nan替换成了0）
# y_fitted, y_hat = moving_average.moving_average(y_origin,predict_num, 3)
# 指数平滑模型
# y_fitted, y_hat, alpha, beta, gamma = exponential_smoothing.exponential_smoothing(y_origin, predict_num)
# print(alpha,beta,gamma)
# SARIMA
y_fitted, y_hat = SARIMA.SARIMA(y_origin,predict_num,2,0,0,2,0,0,365)

index1 = list(range(1,731))
index2 = list(range(731,731 + predict_num))

plt.rcParams['font.family'] = 'SimHei'
# # 绘制拟合值、预测值和原始数据的图表
plt.figure(figsize=(10, 6))

# # 原始数据
plt.plot(index1, y_origin, label='原始数据')

# # 拟合值
plt.plot(index1, y_fitted, label='拟合值')
#
# # 预测值
plt.plot(index2, y_hat, label='预测值')

# 求MAE, MSE, MAPE inf被剔除
print(error_analysis.getError(y_origin,y_fitted))


plt.xlabel('时间')
plt.ylabel('货运量')
plt.title('拟合值、预测值和原始数据')
plt.legend()
plt.show()


