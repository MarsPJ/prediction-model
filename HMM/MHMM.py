"""
MultinomialHMM类是用于处理离散观测符号的隐马尔可夫模型。它适用于离散观测符号的序列，比如文本数据中的单词、声音信号中的音频符号等
"""
"""
n_components：表示隐马尔可夫模型中的隐藏状态数量，也就是在这个例子中，有多少个隐藏状态（E1、E2和E3）。

n_iter：表示在拟合模型时使用的期望最大化（Expectation-Maximization）算法的迭代次数。EM算法用于估计HMM中的参数。

tol：表示EM算法的收敛阈值。当对数似然函数的变化小于该阈值时，认为EM算法已经收敛并停止迭代。
"""