import numpy as np
import d0.functions as F
from d0.variable import Variable


if __name__ == '__main__':
    A = F.Square()
    B = F.Exp()
    C = F.Square()
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    print(y)

    y.grad = np.array(1.0)
    b.grad = C.backward(y.grad)
    a.grad = B.backward(b.grad)
    x.grad = A.backward(a.grad)
    print(x.grad)
