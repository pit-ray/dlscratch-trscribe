import numpy as np

import d0
from d0 import \
        Function, \
        Variable, \
        as_variable, \
        as_array, \
        cuda


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
        xp = cuda.get_array_module(x)
        return xp.exp(x)

    def backward(self, gy):
        x, = self.inputs
        gx = exp(x) * gy
        return gx


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        return gy * cos(x)


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
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
        xp = cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
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


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs

        gb = sum_to(gy, b.shape) if (b.data is not None) else None
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None  # Save memory
    return y


class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = 1 / (1 + xp.exp(-x))
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)


def sigmoid_simple(x):
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y


class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x):
    return ReLU()(x)


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        return get_item_grad(gy, self.slices, x.shape)


def get_item(x, slices):
    return GetItem(slices)(x)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape)
        xp.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item_grad(x, slices, in_shape):
    return GetItemGrad(slices, in_shape)(x)


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.exp(x - x.max(axis=self.axis, keepdims=True))
        return y / y.sum(axis=self.axis, keepdims=True)

    def backward(self, gy):
        y = self.outputs[0]()
        gt = y * gy
        gx = gt - y * gt.sum(axis=self.axis, keepdims=True)
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


def softmax_simple(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    return y / sum(y, axis=axis, keepdims=True)


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        xp = cuda.get_array_module(x)

        N = x.shape[0]

        max_v = x.max(axis=1, keepdims=True)

        diff = x - max_v
        s = xp.exp(diff).sum(axis=1, keepdims=True)
        log_p = diff - xp.log(s)

        t_log_p = log_p[xp.arange(N), t.ravel()]
        y = -t_log_p.sum() / xp.float32(N)
        return y

    def backward(self, gy):
        xp = cuda.get_array_module(gy.data)

        x, t = self.inputs
        N, class_num = x.shape

        y = softmax(x)
        t_onehot = xp.eye(class_num, dtype=t.dtype)[t.data]
        gx = (y - t_onehot) * gy / N
        return gx


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


def softmax_cross_entropy_simple(x, t):
    xp = cuda.get_array_module(x)

    x, t = as_variable(x), as_variable(t)

    N = x.shape[0]

    p = softmax(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p[xp.arange(N), t.data]
    y = -1 * sum(tlog_p) / N

    return y


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        # This forward graph is equal to
        # clip(x) = (x * mask) + [x_min or x_max] * (1 - mask)
        # clip'(x) = gy * mask
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


def accuracy(y, t):
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = as_array((pred == t.data).mean())
    return Variable(result)


def dropout(x, drop_ratio=0.5):
    x = as_variable(x)
    if d0.Config.train:
        xp = cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > drop_ratio
        scale = 1.0 - drop_ratio
        y = x * mask / scale
        return y
    return x


class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        raise NotImplementedError()
        return gy


def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)


from d0.im2col import im2col
from d0.conv import conv2d, max_pool2d
