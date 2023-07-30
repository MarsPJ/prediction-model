function dy=df1(x,y)
    dy = zeros(3,1);% 初始化
    dy(1)=y(2)*y(3);
    dy(2)=-y(1)*y(3);
    dy(3)=-0.51*y(1)*y(2);
end