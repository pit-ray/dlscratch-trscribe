import numpy as np
from d0.variable import Variable


if __name__ == '__main__':
    data = np.array(1.0)
    x = Variable(data)
    print(x.data)

    x.data = np.array(2.0)
    print(x.data)
