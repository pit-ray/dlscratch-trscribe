import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math

import numpy as np
from d0 import Variable
from d0 import functions as F


def rosenblock(x0, x1, a=1,  b=100):
    y = b * (x1 - x0 ** 2) ** 2 + (a - x0) ** 2
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

lr = 0.001
iteration = 10000

for i in range(iteration):
    print(x0, x1)

    y = rosenblock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    # SGD
    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad
