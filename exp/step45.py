import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from d0 import Variable, Model
from d0 import functions as F
from d0 import layers as L
from d0.models import MLP

import matplotlib.pyplot as plt


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


def case1():
    x = Variable(np.random.randn(5, 10), name='x')
    model = TwoLayerNet(100, 10)
    model.plot(x)


def case2():
    np.random.seed(0)

    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    x, y = Variable(x), Variable(y)

    hidden_size = 10
    # model = TwoLayerNet(hidden_size, 1)
    model = MLP((hidden_size, 1))

    lr = 0.2
    iteration = 10000

    for i in range(iteration):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        for p in model.params():
            p.data -= lr * p.grad.data

        if i % 1000 == 0:
            print(loss)

    plt.scatter(x.data, y.data, s=10)

    x_axis = np.linspace(0, 1, 100)
    plt.plot(x_axis, model(x_axis.reshape(100, 1)).data, color='coral')
    plt.show()

# case1()
case2()
