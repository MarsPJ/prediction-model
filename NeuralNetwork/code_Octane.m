load data_Octane.mat
% 1.在Matlab的菜单栏点击APP，再点击Neural Fitting app.
% 2.记得要保存训练的模型net（右键另存为）
% 3.结果中：
% net为生成的神经网络模型
% output为拟合值
% error为残差（原始值-拟合值）


% 4.利用训练出来的神经网络模型对数据进行预测
% 例如我们要预测编号为51的样本，其对应的401个吸光度为：new_X(1,:)
% 这里要注意，我们要将指标变为列向量，然后再用sim函数预测
sim(net, new_X(1,:)')
%
% 写一个循环，预测接下来的十个样本的辛烷值
predict_y = zeros(10,1); % 初始化predict_y
for i = 1: 10
    result = sim(net, new_X(i,:)');
    predict_y(i) = result;
end
disp('预测值为：')
disp(predict_y)