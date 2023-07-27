import numpy as np
import pandas as pd
import HMM

file_path = './data.xlsx'
data = pd.read_excel(file_path)
# 状态类型
states = ['E1', 'E2', 'E3']

# 获取状态概率转移矩阵
transition_matrix = HMM.generate_transition_matrix(data["状态"], states)
print("状态概率转移矩阵")
print(transition_matrix)

# 状态概率转移矩阵可视化
HMM.draw_transition_matrix(transition_matrix, states)

# 获取终极状态概率
steady_states = HMM.get_steady_states(transition_matrix)
print("终极状态概率：")
print(steady_states)

# 预测
predict_num = 11
initial_state = np.array([0, 1, 0])  # 初始状态（最后一个样本）为E2
ret = HMM.HMM(initial_state, transition_matrix, predict_num, states)
print("预测结果为：")
print(ret)

HMM.draw_directed_graph(states, transition_matrix)
