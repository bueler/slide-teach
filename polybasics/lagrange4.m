% LAGRANGE4  Plot four Lagrange polynomials.

n = 4;  x = [-1 0 3 5]';  % just the x-coordinates of the points

xf = -2:0.01:6;  % fine grid of points for plotting
ll = ones(4,length(xf));
for i=1:4
  for j=1:4
    if j ~= i  % skip case where j == i
      ll(i,:) = ll(i,:) .* (xf - x(j)) / (x(i) - x(j));
    end
  end
end

plot(xf,ll(1,:),xf,ll(2,:),xf,ll(3,:),xf,ll(4,:))
legend('l_1(x)','l_2(x)','l_3(x)','l_4(x)')
hold on
plot(x(1),1,'o','markersize',12,...
     x(2),1,'o','markersize',12,...
     x(3),1,'o','markersize',12,...
     x(4),1,'o','markersize',12)
plot(x,[0 0 0 0],'ok','markersize',12)
hold off, grid on

