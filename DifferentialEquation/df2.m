function dy=df2(x,y)
    dy = zeros(2,1);
    dy(1) = y(2);
    dy(2) = 2*x*y(2)/(1+x^2);
end