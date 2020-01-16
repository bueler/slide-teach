function richardspect(A1,A2,str1,str2)
% RICHARDSPECT  Generate figure showing spectral radius of
%   I - omega A
% for -5 <= omega <= 5 and the two input matrices A1,A2.
% Figure label uses str1,str2.  Example:
%   >> [A1,A2] = generateLS;
%   >> richardspect(A1,A1,'LS1','LS2')
%   >> print -dpdf richardspect.pdf

A = {A1, A2};
omega = -1:.01:1;
rho = zeros(2,length(omega));
for i = 1:2
   for j = 1:length(omega)
       M = eye(size(A{i})) - omega(j) * A{i};
       rho(i,j) = max(abs(eig(M)));
   end
end
plot(omega,rho(1,:),omega,rho(2,:))
legend(str1,str2)
grid on
axis tight
axis([omega(1) omega(end) 0 2.2])
xlabel('omega')
ylabel('rho(I - omega A)')

