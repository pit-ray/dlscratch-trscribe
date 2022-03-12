import numpy as np
from d0.variable import Variable
from d0 import functions as F


x = Variable(np.array(2.0))
a = F.square(x)
y = F.add(F.square(a), F.square(a))
y.backward()

print(y.data)
print(x.grad)
