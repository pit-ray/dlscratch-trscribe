import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image

from d0.datasets import Dataset, SinCurve
from d0.models import VGG16, Model
from d0 import utils, test_mode
from d0 import layers as L
from d0 import functions as F
from d0 import optimizers, no_grad
import matplotlib.pyplot as plt


def test1():
    rnn = L.RNN(10)
    x = np.random.randn(1, 1)
    h = rnn(x)
    print(h.shape)


class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y


def test2():
    seq_data = [np.random.randn(1, 1) for _ in range(1000)]
    xs = seq_data[:-1]
    ts = seq_data[1:]

    model = SimpleRNN(10, 1)

    loss, cnt = 0, 0
    for x, t in zip(xs, ts):
        y = model(x)
        loss += F.mean_squared_error(y, t)

        cnt += 1
        if cnt == 2:
            model.cleargrads()
            loss.backward()
            break


def test3():
    train_set = SinCurve(train=True)
    print(len(train_set))
    print(train_set[0])
    print(train_set[1])
    print(train_set[2])

    xs = [example[0] for example in train_set]
    ts = [example[1] for example in train_set]

    plt.plot(np.arange(len(xs)), xs, label='xs')
    plt.plot(np.arange(len(ts)), ts, label='ts')
    plt.show()


def test4():
    np.random.seed(0)

    max_epoch = 100
    hidden_size = 100
    bptt_length = 30

    train_set = SinCurve(train=True)
    seqlen = len(train_set)

    model = SimpleRNN(hidden_size, 1)
    optimizer = optimizers.MomentumSGD().setup(model)

    for epoch in range(max_epoch):
        model.reset_state()
        loss, count = 0, 0

        losses = []
        for x, t in train_set:
            x = x.reshape(1, 1)
            y = model(x)
            # loss += F.mean_squared_error(y, t)  # overflow?
            loss += F.mean_squared_error(y, t) / bptt_length
            count += 1

            if count % bptt_length == 0 or count == seqlen:
                model.cleargrads()
                loss.backward()
                loss.unchain_backward()
                losses.append(float(loss.data))
                loss = 0
                optimizer.update()

        print('| epoch: {}/{} | loss {:.6f}'.format(
            epoch + 1, max_epoch, np.array(losses).mean()))

    xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
    model.reset_state()
    pred_list = []

    with no_grad():
        for x in xs:
            x = np.array(x).reshape(1, 1)
            y = model(x)
            pred_list.append(float(y.data))

    plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
    plt.plot(np.arange(len(xs)), pred_list, label='predict')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


test4()
