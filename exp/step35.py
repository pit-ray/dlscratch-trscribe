import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from d0 import Variable
from d0 import functions as F

from d0.utils import plot_dot_graph


x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.anme = 'y'
y.backward(create_graph=True)

iteration = 1

for i in range(iteration):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx = x.grad
gx.name = 'gx' + str(iteration + 1)
plot_dot_graph(gx, 'tanh.png', verbose=False)
