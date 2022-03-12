import numpy as np

from d0 import functions as F
from d0.variable import Variable
from d0.config import using_config, no_grad


x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))

t = F.add(x0, x1)
y = F.add(x0, t)
y.backward()

print(y.grad, t.grad)
print(x0.grad, x1.grad)


# with using_config('enable_backprop', False):
with no_grad():
    x = Variable(np.array(2.0))
    y = F.square(x)
    print(y.data)
