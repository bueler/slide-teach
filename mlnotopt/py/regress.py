import numpy as np
from numpy.random import uniform, normal
from numpy.linalg import solve
import matplotlib.pyplot as plt

# generate N points once and for all, with random deviations
# from  y = C[0] + C[1]*x + C[2]*x**2
N = 100
C = [1.0, 1.5, -0.2]
sig = 0.4
x = normal(loc=5.0, scale=2.0, size=N)
y = C[0] + C[1] * x + C[2] * x**2 + normal(loc=0.0, scale=sig, size=N)
print(f'true:        {C}')

def boxfig():
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0.0, 10.0)
    plt.ylim(-5.0, 5.0)
    plt.grid(True)

# Figure 1 and Figure 2:  normal equations to do full regression
#   A c = y  -->  A.T A c = A.T y
A = np.array([np.ones((N,)), x, x**2]).T  # goofy way to get columns
#print(f'A is {np.shape(A)[0]} x {np.shape(A)[1]}')
c = solve(np.matmul(A.T, A), np.matmul(A.T, y))
print(f'regression:  {c}')
plt.figure(1)
plt.plot(x, y, 'bo')
boxfig()
plt.show()
plt.figure(2)
plt.plot(x, y, 'bo')
xx = np.linspace(0.0, 10.0, num=201)
yy = c[0] + c[1]*xx + c[2]*xx**2
plt.plot(xx, yy, 'r', label='fit all data')
plt.legend(loc='lower left')
boxfig()
plt.show()

# Figures 3 .. 6: show that fully-fitting batch is unstable
for j in range(6):
    mb = 4   # batch size
    xb = x[mb*j:mb*(j+1)]
    yb = y[mb*j:mb*(j+1)]
    Ab = np.array([np.ones((mb,)), xb, xb**2]).T
    cb = solve(np.matmul(Ab.T, Ab), np.matmul(Ab.T, yb))
    print(f'batch {j:2d}:    {cb}')
    plt.figure(3 + j)
    plt.plot(x, y, 'bo')
    plt.plot(xb, yb, 'go', ms=12, label=f'batch {j}')
    plt.plot(xx, yy, 'r:')
    yyb = cb[0] + cb[1] * xx + cb[2] * xx**2
    plt.plot(xx, yyb, 'g')
    plt.legend(loc='lower left')
    boxfig()
    plt.show()
