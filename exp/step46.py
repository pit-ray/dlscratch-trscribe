import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from d0 import Variable, Model
from d0 import functions as F
from d0.models import MLP
from d0 import optimizers

import matplotlib.pyplot as plt


np.random.seed(0)

x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

x, y = Variable(x), Variable(y)

lr = 0.2
iteration = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
# optimizer = optimizers.SGD(lr=lr)
optimizer = optimizers.MomentumSGD(lr=lr)
optimizer.setup(model)

for i in range(iteration):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()

    if i % 1000 == 0:
        print(loss)

plt.scatter(x.data, y.data, s=10)

x_axis = np.linspace(0, 1, 100)
plt.plot(x_axis, model(x_axis.reshape(100, 1)).data, color='coral')
plt.show()
