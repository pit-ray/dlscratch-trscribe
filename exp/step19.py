import numpy as np

from d0 import functions as F
from d0.variable import Variable


x = Variable(np.array([1, 2, 3]))
print(x)

x = Variable(None)
print(x)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
print(x)
