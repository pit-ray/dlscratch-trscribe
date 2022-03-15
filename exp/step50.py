import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import d0
from d0 import DataLoader
from d0.datasets import Spiral
from d0.models import MLP
from d0.optimizers import SGD
from d0 import functions as F

import matplotlib.pyplot as plt


batch_size = 30
max_epoch = 300
hidden_size = 10
lr = 1.0

train_set = Spiral(train=True)
test_set = Spiral(train=False)

train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = SGD(lr=lr).setup(model)

epoch_losses = {'train': [], 'test': []}
epoch_accs = {'train': [], 'test': []}

for epoch in range(max_epoch):
    # Train
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

    print('epoch: {}/{}'.format(epoch + 1, max_epoch))

    avg_loss = np.array(losses).mean()
    avg_acc = np.array(accs).mean()

    epoch_losses['train'].append(avg_loss)
    epoch_accs['train'].append(avg_acc)
    print('train loss: {:.4f}, accuracy: {:.4f}'.format(avg_loss, avg_acc))

    # Validation
    losses = []
    accs = []
    with d0.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)

            losses.append(float(loss.data))
            accs.append(float(acc.data))

    avg_loss = np.array(losses).mean()
    avg_acc = np.array(accs).mean()

    epoch_losses['test'].append(avg_loss)
    epoch_accs['test'].append(avg_acc)
    print('test loss: {:.4f}, accuracy: {:.4f}'.format(avg_loss, avg_acc))


fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.legend(loc='upper left')

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
ax2.legend(loc='lower right')

epochs = np.arange(max_epoch)

for phase in ['train', 'test']:
    ax1.plot(epochs, epoch_losses[phase], label=phase)
    ax2.plot(epochs, epoch_accs[phase], label=phase)

fig.tight_layout()
plt.show()
