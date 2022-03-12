import numpy as np

from d0.variable import Variable
import d0.functions as F


if __name__ == '__main__':
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    y = F.add(x0, x1)
    print(y.data)
