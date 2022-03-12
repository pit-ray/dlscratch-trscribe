import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math

import numpy as np
from d0 import Variable


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


def gx2(x):
    return 12 * x ** 2 - 4


x = Variable(np.array(2.0))
iteration = 10

for i in range(iteration):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward()

    x.data -= x.grad / gx2(x.data)
