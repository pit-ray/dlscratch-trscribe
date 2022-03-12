import numpy as np


def add_backward(x_grad, gx):
    print('gx: {}({}), x.grad: {}({})'.format(gx, id(gx), x, id(x)))
    x_grad += gx
    print('gx: {}({}), x.grad: {}({})'.format(gx, id(gx), x, id(x)))


if __name__ == '__main__':
    gx = np.array(1.0)
    x = np.array(3.0)

    add_backward(x, gx)
    add_backward(x, gx)
