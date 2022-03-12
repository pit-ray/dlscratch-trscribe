import numpy as np
from d0.variable import Variable
import d0.functions as F


if __name__ == '__main__':
    x = Variable(np.array(0.5))
    y = F.square(F.exp(F.square(x)))
    y.backward()
    print(x.grad)

    Variable(1.0)
