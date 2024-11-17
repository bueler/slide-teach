import numpy as np
from numpy.random import uniform, normal
from numpy.linalg import norm, solve
import matplotlib.pyplot as plt

regenerate = False
datafile = 'points.npy'

# model  y = C[0] + C[1]*x + C[2]*x**2
ctrue = [1.0, 1.5, -0.2]
print(f'true:        {ctrue}')

# generate or load N points, with random deviations from model
if regenerate:
    N = 100
    sig = 0.4
    x = normal(loc=5.0, scale=2.0, size=N)
    y = C[0] + C[1] * x + C[2] * x**2 + normal(loc=0.0, scale=sig, size=N)
    with open(datafile, 'wb') as f:
        np.save(f, x)
        np.save(f, y)
else:
    with open(datafile, 'rb') as f:
        x = np.load(f)
        y = np.load(f)
    N = len(x)
    if len(y) != N:
        raise ValueError

def finishfig(name, prefix='output/'):
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0.0, 10.0)
    plt.ylim(-5.0, 5.0)
    plt.grid(True)
    import os
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    out = prefix + name + '.png'
    print(f'   writing figure {out} ...')
    plt.savefig(out, bbox_inches='tight')
    plt.close()

# objective
def f(c, x, y):
    K = len(x)
    A = np.array([np.ones((K,)), x, x**2]).T
    return 0.5 * np.norm(np.matmul(A, c) - y)**2

# gradient
def gradf(c, x, y):
    K = len(x)
    A = np.array([np.ones((K,)), x, x**2]).T
    return np.matmul(np.matmul(A.T, A), c) - np.matmul(A.T, y)

# solve normal equations:
#   "A c = y"  -->  A.T A c = A.T y  -->  c = (A.T A) \ (A.T y)
def normal(x, y):
    K = len(x)
    A = np.array([np.ones((K,)), x, x**2]).T  # goofy way to get columns
    c = solve(np.matmul(A.T, A), np.matmul(A.T, y))
    return c

# regression on full data
c = normal(x, y)
print(f'full fit:  {c}')
plt.figure()
plt.plot(x, y, 'bo')
xx = np.linspace(0.0, 10.0, num=201)
yy = c[0] + c[1]*xx + c[2]*xx**2
plt.plot(xx, yy, 'r')
finishfig('full')

# split data in half and "validate"
hN = int(N / 2)
c1 = normal(x[:hN], y[:hN])
print(f'1st half fit:  {c1}')
c2 = normal(x[hN:], y[hN:])
print(f'2nd half fit:  {c2}')
plt.figure()
plt.plot(x, y, 'bo')
xx = np.linspace(0.0, 10.0, num=201)
yy1 = c1[0] + c1[1]*xx + c1[2]*xx**2
yy2 = c2[0] + c2[1]*xx + c2[2]*xx**2
plt.plot(xx, yy1, 'r', label=f'fit 1 .. {hN}')
plt.plot(xx, yy2, 'g', label=f'fit {hN+1} .. {N}')
plt.legend()
finishfig('valid')

# mode 0: fully-fitting batches (is unstable)
# mode 1: steps of SGD with fixed learning rate
roots = ['batch', 'sgd']
steps = [6, 2000]
skip = [1, 99]
mb = 4   # batch size
lam = 0.001 / mb
xcat = np.concatenate((x, x)) # repeat arrays for periodic batching
ycat = np.concatenate((y, y))
for mode in range(2):
    if mode == 1:
        cb = np.array([1.0, 1.0, 0.0])
        #cb = ctrue.copy()
    for j in range(steps[mode]):
        js, je = np.mod(mb*j, N), np.mod(mb*(j+1), N)
        xb, yb = xcat[js:je], ycat[js:je]
        if mode == 0:
            cb = normal(xb, yb)  # fully solve for c
        else:
            g = gradf(cb, xb, yb)
            cb -= lam * g  # SGD step for c
        print(f'{roots[mode]} {j+1:2d}:    {cb}')
        if np.mod(j, skip[mode]) == 0:
            plt.figure()
            plt.plot(x, y, 'bo')
            plt.plot(xb, yb, 'go', ms=12, label=f'{roots[mode]} {j+1}')
            plt.plot(xx, yy, 'r:')
            yyb = cb[0] + cb[1] * xx + cb[2] * xx**2
            plt.plot(xx, yyb, 'g')
            plt.legend(loc='lower left')
            finishfig(f'{roots[mode]}{j+1}')
