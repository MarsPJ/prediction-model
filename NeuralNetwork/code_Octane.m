load data_Octane.mat
% 1.��Matlab�Ĳ˵������APP���ٵ��Neural Fitting app.
% 2.�ǵ�Ҫ����ѵ����ģ��net���Ҽ����Ϊ��
% 3.����У�
% netΪ���ɵ�������ģ��
% outputΪ���ֵ
% errorΪ�вԭʼֵ-���ֵ��


% 4.����ѵ��������������ģ�Ͷ����ݽ���Ԥ��
% ��������ҪԤ����Ϊ51�����������Ӧ��401�������Ϊ��new_X(1,:)
% ����Ҫע�⣬����Ҫ��ָ���Ϊ��������Ȼ������sim����Ԥ��
sim(net, new_X(1,:)')
%
% дһ��ѭ����Ԥ���������ʮ������������ֵ
predict_y = zeros(10,1); % ��ʼ��predict_y
for i = 1: 10
    result = sim(net, new_X(i,:)');
    predict_y(i) = result;
end
disp('Ԥ��ֵΪ��')
disp(predict_y)