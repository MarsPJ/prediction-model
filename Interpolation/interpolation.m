% 三次样条插值和分段三次埃尔米特插值的对比
x = -pi:pi; 
y = sin(x); 
% 实际应用时nex_x应该是要插入的x，如：new_x = [1.2,2.5,3.0];
new_x = -pi:0.5:pi;% 
p1 = pchip(x,y,new_x);   %分段三次埃尔米特插值
p2 = spline(x,y,new_x);  %三次样条插值
figure(2);
plot(x,y,'o',new_x,p1,'r-',new_x,p2,'b-')
legend('样本点','三次埃尔米特插值','三次样条插值','Location','SouthEast')   %标注显示在东南方向
% 说明：
% LEGEND(string1,string2,string3, …)
% 分别将字符串1、字符串2、字符串3……标注到图中，每个字符串对应的图标为画图时的图标。
% ‘Location’用来指定标注显示的位置


% n维数据的插值
x = -pi:pi; 
y = sin(x); 
new_x = [1.2,2.5,3.0];
p = interpn (x, y, new_x, 'spline');
% 等价于 p = spline(x, y, new_x);
figure(3);
plot(x, y, 'o', new_x, p, '*')