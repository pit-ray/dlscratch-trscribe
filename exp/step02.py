import numpy as np
from d0.variable import Variable
from d0.functions import Square


if __name__ == '__main__':
    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    print(type(y))
    print(y.data)
