import numpy as np
import matplotlib.pyplot as plt


def generate_transition_matrix(y_origin, states):
    # 统计从Ei 到 Ej 的转换次数
    transitions = {}
    for i in range(len(states)):
        for j in range(len(states)):
            transitions[states[i]] = {x:0 for x in states}

    # transitions = {'E1': {'E1': 0, 'E2': 0, 'E3': 0},
    #                'E2': {'E1': 0, 'E2': 0, 'E3': 0},
    #                'E3': {'E1': 0, 'E2': 0, 'E3': 0}}
    # print(transitions)
    for i in range(len(y_origin) - 1):
        current_state = y_origin[i]
        next_state =  y_origin[i + 1]
        transitions[current_state][next_state] += 1

    # 统计Ei 出现的总次数
    state_counts = {x:0 for x in states}
    # print(state_counts)
    # state_counts = {'E1': 0, 'E2': 0, 'E3': 0}

    for state in transitions:
        state_counts[state] = sum(transitions[state].values())

    # 统计初始的状态转移矩阵
    transition_matrix = np.zeros((len(transitions.keys()), len(transitions.keys())))
    for i, state in enumerate(transitions):
        for j, next_state in enumerate(transitions[state]):
            transition_matrix[i, j] = transitions[state][next_state] / state_counts[state]
    return transition_matrix


def HMM(initial_state, transition_matrix, predict_num, states):
    # 4. 定义一个存储结果的 DataFrame，结果与 Ma 相同
    mats1 = np.zeros((predict_num, len(states)))
    """
    在这个循环中，我们通过迭代计算中间状态概率。对于每个 i（0 到 num_steps-1），
    通过 np.linalg.matrix_power(Ma, i) 计算转移概率矩阵 Ma 的 i 次方，然后使用初始状态概率 inital 与该幂次矩阵相乘。
    这将得到每个步骤 i 对应的中间状态概率，然后将其保存在 mats1 数组中。
    """
    current_state = np.dot(initial_state, transition_matrix)
    for i in range(predict_num):
        mats1[i] = current_state
        current_state = np.dot(current_state, transition_matrix)
    # print(mats1)
    ret = []
    for i in range(predict_num):
        index = np.argmax(mats1[i])
        ret.append(states[index])
    return ret


def get_steady_states(transition_matrix):
    """
    使用 np.linalg.eig(Ma.T) 计算转移概率矩阵 Ma 转置后的特征值 eigenvalues 和特征向量 eigenvectors。
    然后，通过 np.argmin(np.abs(eigenvalues - 1.0)) 找到特征值数组中最接近 1.0 的特征值的索引，
    这个索引对应于单位特征值（单位特征值与稳态概率之间存在关系）。
        在隐马尔可夫模型中，终态概率表示在时间趋于无穷大时，模型达到稳态后各个状态的概率分布。
        当模型达到稳态时，状态的概率不再发生变化，这时特征向量对应的特征值就是1.0，称为单位特征值。
        单位特征值与稳态概率之间存在着重要的关系。假设 v 是一个单位特征值对应的特征向量，
        其各个元素表示在稳态时各个状态的概率分布。我们知道，在稳态时，状态概率分布不再改变，
        也就是转移概率矩阵的作用不再改变状态分布。而单位特征值对应的特征向量恰好描述了这个稳态的状态分布。

    然后，取特征向量数组中对应索引的列，这个列就是对应于单位特征值的特征向量。
    最后，将该特征向量取实部，并进行归一化，得到隐马尔可夫模型的终态概率 steady_states。
    """

    # 通过计算转移概率矩阵的特征向量来获取终态概率
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    index = np.argmin(np.abs(eigenvalues - 1.0))
    steady_states = np.real(eigenvectors[:, index])
    steady_states /= np.sum(steady_states)
    return steady_states


def draw_transition_matrix(transition_matrix, states):
    # 3. 转移概率矩阵可视化
    plt.figure(figsize=(8, 6))

    """
    Ma：这是我们要绘制的概率转移矩阵，是一个二维数组。概率转移矩阵是隐马尔可夫模型中定义状态之间转移概率的矩阵。

    interpolation='nearest'：这是指定图像绘制时使用的插值方法。在这里，我们使用'nearest'插值，它表示在像素之间使用最近邻像素的值来填充，这样可以保持概率转移矩阵的离散性。

    cmap='Blues'：这是指定颜色映射（colormap）的名称，用于将数值映射到颜色。在这里，我们使用'Blues'颜色映射，
    它会将较小的数值映射为深蓝色，较大的数值映射为浅蓝色，使我们能够更直观地观察概率转移矩阵的分布。
    """
    plt.imshow(transition_matrix, interpolation='nearest', cmap='Blues')
    plt.xticks(np.arange(len(states)), states)
    plt.yticks(np.arange(len(states)), states)
    plt.colorbar(label='Probability')
    plt.xlabel('To State')
    plt.ylabel('From State')
    plt.title('Transition Matrix')
    plt.show()


import networkx as nx
import matplotlib.pyplot as plt


def draw_directed_graph(states, transition_matrix):
    # 创建有向图对象
    G = nx.DiGraph()

    # 添加状态节点
    G.add_nodes_from(states)

    # 添加转移边和转移次数
    for i, from_state in enumerate(states, 0):
        for j, to_state in enumerate(states, 0):
            G.add_edge(from_state, to_state, weight=round(transition_matrix[i][j], 2))
            print(from_state,to_state,transition_matrix[i][j])

    # 绘制状态转移图
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # 使用spring_layout()并指定seed
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=12, font_weight='bold')

    # 获取边的标签信息
    edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges}
    print(edge_labels)

    # 绘制边的标签
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.title('State Transition Diagram', fontsize=16)
    plt.show()
