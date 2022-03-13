import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from d0 import Variable
from d0 import functions as F


x = Variable(np.random.rand(2, 4, 5))
# y = F.reshape(x, (6,))
# y = x.reshape((6, ))

# y = F.transpose(x)
# y = x.transpose()
# y = x.T
y = x.transpose((1, 0, 2))
print(x.shape, y.shape)

y.backward(retain_grad=True)
print(x.grad)
