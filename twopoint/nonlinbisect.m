function AA = nonlinbisect(Aa,Ab,tol)
% NONLINBISECT   Find A so that the solution to the ODE IVP
%   u'' + u^3 = 0,  u(0) = 1,  u'(0) = A
% solves the ODE BVP
%   u'' + u^3 = 0,  u(0) = 1,  u(1) = -2
% Thus we solve the ODE BVP by shooting.  Uses ODE45 for
% solving the ODE IVPs and bisection to achieve "u(1)=-2".
% Usage:
%   >> A = nonlinbisect(Aa,Ab,tol)
% Example:
%   >> nonlinbisect(-5,5,1e-4);

% ODE as a system is:   u' = v,  v' = -u^3   or  Y' = G(Y)
G = @(x,Y) [Y(2); -Y(1).^3];

% establish and check initial bracket
Fa = getF(Aa,G);
Fb = getF(Ab,G);
if Fa * Fb >= 0,  error('must start with bracket'), end
fprintf('initial bracket:  %.12f <= A <= %.12f\n',Aa,Ab)

% do bisection
maxiter = 100;  % fail if reach this
for j = 1:maxiter
  AA = (Aa + Ab) / 2;
  FF = getF(AA,G);
  if FF * Fb < 0
    Aa = AA;
    Fa = FF;
  else
    Ab = AA;
    Fb = FF;
  end
  fprintf('        bracket:  %.12f <= A <= %.12f\n',Aa,Ab)
  if abs(Ab - Aa) < tol
    break
  end
end
if j == maxiter,  error('maximum number of iterations reached'),  end
end

% return z = u(1) from solving ODE IVP with u'(0) = A
function z = getF(A,G)
  x = 0:0.01:1;
  [xout, Y] = ode45(G,x,[1.0; A]);
  z = Y(end,1) - (-2);
end
