clear;clc
year = 1790:10:2000;
population = [3.9,5.3,7.2,9.6,12.9,17.1,23.2,31.4,38.6,50.2,62.9,76.0,92.0,106.5,123.2,131.7,150.7,179.3,204.0,226.5,251.4,281.4];
cftool  % 拟合工具箱
% (1) X data 选择 year
% (2) Y data 选择 population
% (3) 拟合方式选择：Custom Equation (自定义方程)
% (4) 修改下方的方框为：x = f(t) = xm/(1+(xm/3.9-1)*exp(-r*(t-1790)))
% (5) 左边的result一栏最上面显示：Fit computation did not converge:即没有找到收敛解，右边的拟合图形也表明拟合结果不理想
% (6) 点击Fit Options，修改非线性最小二乘估计法拟合的初始值(StartPoint), r修改为0.02，xm修改为500
% (7) 此时左边的result一览得到了拟合结果：r = 0.02735, xm = 342.4
% (8) 依次点击拟合工具箱的菜单栏最左边的文件—Generate Code(导出代码到时候可以放在你的论文附录)，可以得到一个未命名的脚本文件
% (9) 在这个打开的脚本中按快捷键Ctrl+S，将这个文件保存到当前文件夹。
% (10) 在现在这个文件中调用这个函数得到参数的拟合值和预测的效果
[fitresult, gof] = createFit(year, population) 
%        r =     0.02735  (0.0265, 0.0282)
%        xm =       342.4  (311, 373.8)
r = 0.02735;
xm = 342.4;
t = 2001:1:2030;
hold on
prediction = xm./(1+(xm./3.9-1).*exp(-r.*(t-1790)));
figure(2);
plot(year,population,'o',t,prediction,'.');
