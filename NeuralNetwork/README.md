# 神经网络预测

**需要的参数**

- 原始数据
- 可选参数：
  - 隐藏层神经元个数
  - 训练集、测试集、验证集的比例
  - 选择求解的算法
    - 速度快
      - 莱文贝格-马夸特方法 (Levenberg-Marquardt algorithm）
      - 量化共钜梯度法 (Scaled Conjugate Gradient )
    - 可防止过拟合，但是求解速度慢（清风推荐）
      - 贝叶斯正则化方法 (Bayesian-regularization）