import numpy as np

from d0 import functions as F
from d0.variable import Variable


x = Variable(np.array(2.0))
y = -x
print(y)

y = 2.0 - x
print(y)

y = x - 1.0
print(y)

y = x / 4.0
print(y)

y = 2.0 / x
print(y)

y = x ** 3.0
print(y)
