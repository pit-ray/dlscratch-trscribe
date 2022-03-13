import numpy as np

from d0 import Function, as_variable
from d0 import utils


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0]
        gx = exp(x) * gy
        return gx


def exp(x):
    return Exp()(x)


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        return gy * cos(x)


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    return Tanh()(x)


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        new_axes = tuple(np.argsort(self.axes))
        return transpose(gy, new_axes)


def transpose(x, axes=None):
    return Transpose(axes)(x)


class Sum(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        ndim = len(self.x_shape)

        # self.axis to tuple
        tupled_axis = self.axis
        if self.axis is None:
            tupled_axis = None
        elif not isinstance(self.axis, tuple):
            tupled_axis = (self.axis, )

        # If some axes have been reduced, it restore to the inputted axes.
        if (ndim > 0) and (tupled_axis is not None) and (not self.keepdims):

            # If axis is negative it counts from the last to the first axis.
            actual_axis = [(a if a >= 0 else a + ndim) for a in tupled_axis]
            actual_axis.sort()

            shape = list(gy.shape)
            for a in actual_axis:
                shape.insert(a, 1)  # restore a reduced axis

            gy = gy.reshape(shape)

        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        # Shape with the axis of sum as 1.
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape

        # For example, x.shape = (2, 3, 4, 5), shape = (4, 1)
        ndim = len(self.shape)  # 2
        lead = x.ndim - ndim  # 2 = 4 - 2
        lead_axis = tuple(range(lead))  # (0, 1)

        # (i, sx) = (0, 4), (i + lead) = 2
        # (i, sx) = (1, 1), (i + lead) = 3   <= sx == 1
        # axis = (3)
        axis = tuple([i + lead for i, sx in enumerate(self.shape) if sx == 1])

        # The axes to be reduced and the axes to be added to make 1
        # pass into numpy.sum(), respectively.
        #
        # (lead_axis + axis) = (0, 1, 3)
        # y.shape = (1, 1, 4, 1)
        y = x.sum(lead_axis + axis, keepdims=True)

        if lead > 0:
            y = y.squeeze(lead_axis)  # y.shape = (4, 1)

        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


def matmul(x, W):
    return MatMul()(x, W)


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2.0 / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)
