function [x,U] = bvpqgs(m,xL,xR,q,f,alpha,beta,tol)
% THIS CODE WORKS BUT WAS NOT USED IN THE SOLUTIONS BECAUSE P15d
% WAS CANCELLED
% BVPQGS  Solve the ODE BVP
%   u''(x) + q u(x) = f(x),  u(xL)=alpha,  u(xR)=beta
% using centered finite differences, for given input function f(x).
% Applies the Gauss-Seidel method to solve the linear system;
% see GSTOL subfunction below.  Warning if q >= 0 because matrix
% is not strictly diagonally dominant.
% Compare BVPQ which uses "A\b" to solve the linear system.
% Usage:
%   [x,u] = bvpqgs(m,xL,xR,q,f,alpha,beta)
% Example:  Solve an m=20 unknowns problem for which we know the
% exact solution.
%   >> f = @(x) (pi^2/4^2 + 1) * sin((pi/4)*x) - 1;
%   >> [x,U] = bvpqgs(20,0,2,-1,f,1,0);
%   >> uexact = @(x) 1 - sin((pi/4)*x);
%   >> plot(x,U,'o',x,uexact(x),'-')
%   >> xlabel x,  legend('numerical','exact')

if nargin < 8,  tol = 1.0e-8;  end
% set up grid
h = (xR - xL) / (m+1);
x = xL:h:xR;           % length m+2
size(x)

% assemble linear system  A U = F
A = (-2 + q * h^2) * eye(m);
for j = 1:m-1
    A(j,j+1) = 1;
    A(j+1,j) = 1;
end
A = (1/h^2) * A;
F = f(x(2:m+1))';    % evaluate f at interior grid points
F(1) = F(1) - alpha / h^2;
F(m) = F(m) - beta / h^2;

%% uncomment to compute spectral radius; explains performance
%D = diag(diag(A));  L = -tril(A,-1);  U = -triu(A,+1);
%max(abs(eig((D-L) \ U)))

% solve the linear system
%U = A \ F;           % numerical solution at interior points
U0 = zeros(m,1);
fprintf('solving by Gauss-Seidel ... ')
[U, count] = gstol(A,F,U0,tol);
fprintf('%d iterations\n',count)
U = [alpha U' beta]; % whole solution including boundary vals
end % BVPGS

    function [z, count] = gstol(A,b,x0,tol)
    % GSTOL  Solve A x = b by Gauss-Seidel, doing iterations
    % until norm(A* x - b) <= tol

    m = size(A,1);
    x = x0;
    count = 0;
    while norm(A*x - b) > tol
        count = count + 1;
        for i = 1:m
            noti = [1:i-1, i+1:m];
            x(i) = (b(i) - A(i,noti) * x(noti)) / A(i,i);
        end
    end
    z = x;
    end % GSTOL
