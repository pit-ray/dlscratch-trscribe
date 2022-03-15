import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np

from d0 import optimizers
from d0 import datasets
from d0 import functions as F
from d0.models import MLP

import matplotlib.pyplot as plt


max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

np.random.seed(0)

train_set = datasets.Spiral(train=True)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iteration = math.ceil(data_size / batch_size)

losses = []

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iteration):
        batch_index = index[i * batch_size:(i + 1) * batch_size]

        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data)

    avg_loss = sum_loss / max_iteration
    print('epoch {}, loss {:.2f}'.format(epoch + 1, avg_loss))
    losses.append(avg_loss)

# plt.plot(range(max_epoch), losses)
# plt.show()

t_colors = ['b', 'g', 'r']
b_colors = ['c', 'm', 'y']

step = 24

# x_min, x_max = x[:, 0].min(), x[:, 0].max()
# y_min, y_max = x[:, 1].min(), x[:, 1].max()
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0

for pos_y in np.arange(y_min, y_max, (y_max - y_min) / step):
    for pos_x in np.arange(x_min, x_max, (x_max - x_min) / step):
        y = model(np.array([[pos_x, pos_y]]))
        pred_class = np.argmax(y.data)

        plt.scatter([pos_x], [pos_y], c=b_colors[pred_class], s=300)

plt.show()
