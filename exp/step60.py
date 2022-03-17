import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image

from d0.datasets import Dataset, SinCurve
from d0.dataloaders import SeqDataLoader
from d0.models import VGG16, Model
from d0 import utils, test_mode
from d0 import layers as L
from d0 import functions as F
from d0 import optimizers, no_grad
import matplotlib.pyplot as plt


np.random.seed(0)


def test1():
    train_set = SinCurve(train=True)
    dataloader = SeqDataLoader(train_set, batch_size=3)
    x, t = next(dataloader)
    print(x)
    print('-' * 10)
    print(t)


class BetterRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.LSTM(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y


def test2():
    np.random.seed(0)

    max_epoch = 100
    batch_size = 30
    hidden_size = 100
    bptt_length = 30

    train_set = SinCurve(train=True)
    dataloader = SeqDataLoader(train_set, batch_size=batch_size)
    seqlen = len(train_set)

    model = BetterRNN(hidden_size, 1)
    optimizer = optimizers.MomentumSGD().setup(model)

    for epoch in range(max_epoch):
        model.reset_state()
        loss, count = 0, 0

        losses = []
        for i, (x, t) in enumerate(dataloader):
            y = model(x)
            loss += F.mean_squared_error(y, t)
            count += 1

            if count % bptt_length == 0 or count == seqlen:
                # utils.plot_dot_graph(loss, os.path.join('.dezero', 'LSTM.png'))

                model.cleargrads()
                loss.backward()
                loss.unchain_backward()
                losses.append(float(loss.data))
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

test2()
