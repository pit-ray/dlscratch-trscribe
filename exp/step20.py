import numpy as np

from d0 import functions as F
from d0.variable import Variable


a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

# y = F.add(F.mul(a, b), c)

y = a * b + c

y.backward()

print(y)
print(a.grad)
print(b.grad)
