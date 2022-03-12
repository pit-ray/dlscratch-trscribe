import numpy as np

from d0 import functions as F
from d0.variable import Variable


x = Variable(np.array(3.0))
y = F.add(F.add(x, x), x)
y.backward()
print(x.grad, y.grad)
