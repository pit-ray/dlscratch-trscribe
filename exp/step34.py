import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from d0 import Variable
from d0 import functions as F

import matplotlib.pyplot as plt


x = Variable(np.linspace(-7, 7, 200))
y = F.sin(x)
y.backward(create_graph=True)

logs = [y.data.flatten()]

for i in range(3):
    logs.append(x.grad.data.flatten())
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)
    print(x.grad)

labels = ["y=sin(x)", "y'", "y''", "y'''"]
for i, v in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])
plt.legend(loc='lower right')
plt.show()
