import numpy as np

from d0 import functions as F
from d0.variable import Variable


x = Variable(np.array(2.0))
y = x + np.array(3.0)
print(y)


y = 3.0 * x + 1.0
print(y)
