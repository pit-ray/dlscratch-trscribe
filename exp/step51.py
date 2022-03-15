import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from d0 import datasets, optimizers, no_grad
from d0 import DataLoader
from d0.models import MLP
from d0 import functions as F

import matplotlib.pyplot as plt


np.random.seed(0)


def preprocess(x):
    x = x.flatten()
    x = x.astype(np.float32)
    x /= 255.0
    return x


def t1():
    train_set = datasets.MNIST(train=True, transform=preprocess)
    test_set = datasets.MNIST(train=False, transform=preprocess)

    print(len(train_set))
    print(len(test_set))

    x, t = train_set[0]
    print(type(x), x.shape)
    print(t)

    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()


def train(max_epoch=5, batch_size=100, hidden_size=1000):
    train_set = datasets.MNIST(train=True, transform=preprocess)
    test_set = datasets.MNIST(train=False, transform=preprocess)

    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
    optimizer = optimizers.SGD().setup(model)

    for epoch in range(max_epoch):
        print('-' * 20)
        print('epoch: {}/{}'.format(epoch + 1, max_epoch))

        losses = []
        accs = []
        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)

            model.cleargrads()
            loss.backward()
            optimizer.update()

            losses.append(float(loss.data))
            accs.append(float(acc.data))

        print('train loss: {:.4f}, accuracy: {:.4f}'.format(
            np.array(losses).mean(), np.array(accs).mean()))

        losses = []
        accs = []
        with no_grad():
            for x, t in test_loader:
                y = model(x)
                loss = F.softmax_cross_entropy(y, t)
                acc = F.accuracy(y, t)

                losses.append(float(loss.data))
                accs.append(float(acc.data))

        print('test loss: {:.4f}, accuracy: {:.4f}'.format(
            np.array(losses).mean(), np.array(accs).mean()))


train()
