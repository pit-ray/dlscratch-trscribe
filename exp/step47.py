import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from d0 import Variable, Model, as_variable
from d0 import functions as F
from d0.models import MLP
from d0 import optimizers

import matplotlib.pyplot as plt


def softmax1d(x):
    x = as_variable(x)
    y = F.exp(x - np.max(x.data))
    return y / F.sum(y)

np.random.seed(0)

model = MLP((10, 3))

# x = Variable(np.array([[0.2, -0.4]]))
x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
x = Variable(x)
t = np.array([1, 0, 1, 0])
y = model(x)
# p = softmax1d(y)
loss = F.softmax_cross_entropy(x, t)
#loss = F.softmax_cross_entropy_simple(x, t)
model.cleargrads()
loss.backward()
print(x.grad)
