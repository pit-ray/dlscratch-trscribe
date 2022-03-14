import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from d0 import Variable
from d0 import functions as F
from d0 import layers as L

import matplotlib.pyplot as plt


np.random.seed(0)

x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

x, y = Variable(x), Variable(y)

l1 = L.Linear(10)
l2 = L.Linear(1)


def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y


lr = 0.2
iteration = 10000

for i in range(iteration):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()

    loss.backward()

    for layer in [l1, l2]:
        for p in layer.params():
            p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(loss)


plt.scatter(x.data, y.data, s=10)

x_axis = np.linspace(0, 1, 100)
plt.plot(x_axis, predict(x_axis.reshape(100, 1)).data, color='coral')
plt.show()
