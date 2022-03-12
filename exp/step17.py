import numpy as np

from d0 import functions as F
from d0.variable import Variable


for i in range(10):
    x = Variable(np.random.randn(10000))
    y = F.square(F.square(F.square(x)))
