import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math

import numpy as np
from d0 import Variable
from d0 import functions as F

from d0.utils import plot_dot_graph


x = Variable(np.array(np.pi / 4))
y = F.sin(x)
y.backward()

print(y.data)
print(x.grad)

x.cleargrad()


def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break

    return y


y = my_sin(x, threshold=1e-5)
y.backward()
print(y.data)
print(x.grad)

plot_dot_graph(y, 'taylor_sin.png', verbose=False)
