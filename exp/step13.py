import numpy as np

from d0.variable import Variable
import d0.functions as F


if __name__ == '__main__':
    x = Variable(np.array(2.0))
    y = Variable(np.array(3.0))

    z = F.add(F.square(x), F.square(y))
    z.backward()

    print(z.data)
    print(x.grad)
    print(y.grad)
