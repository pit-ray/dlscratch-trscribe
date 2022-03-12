import numpy as np
from d0.variable import Variable
import d0.functions as F
from d0.utils import numerical_diff


def combined_f(x):
    A = F.Square()
    B = F.Exp()
    C = F.Square()
    return C(B(A(x)))


if __name__ == '__main__':
    f = F.Square()
    x = Variable(np.array(2.0))
    dy = numerical_diff(f, x)
    print(dy)

    x = Variable(np.array(0.5))
    dy = numerical_diff(combined_f, x)
    print(dy)
