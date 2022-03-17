import contextlib
import weakref

import numpy as np

import d0


class Config:
    enable_backprop = True
    train = True


@contextlib.contextmanager
def using_config(name, value):
    try:
        old_value = getattr(Config, name)
        setattr(Config, name, value)

        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config('enable_backprop', False)


def test_mode():
    return using_config('train', False)


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, d0.cuda.get_array_types()):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        return 'variable({})'.format(
            str(self.data).replace('\n', '\n' + ' ' * 9))

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            xp = d0.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs, )

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        # If use `x.grad += gx` syntax, the in-place process of
                        # ndarray is performed and the calculation is done
                        # against the memory referred with the left-hand-side
                        # variable. Therefore, the add calculation effects to
                        # the memory of `gx` because `x.grad` has the same
                        # memory as `gx` at `x.grad = gx`.
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

                    if not retain_grad:
                        for y in f.outputs:
                            y().grad = None  # Remove unused grad

    def unchain(self):
        self.creator = None

    def unchain_backward(self):
        if self.creator is None:
            return

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            for x in f.inputs:
                if x.creator is not None:
                    funcs.append(x.creator)
                    x.unchain()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return d0.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)):
                axes = axes[0]
            elif axes[0] is None:
                axes = None
        return d0.functions.transpose(self, axes)

    @property
    def T(self):
        return d0.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return d0.functions.sum(self, axis, keepdims)

    def max(self, axis=None, keepdims=False):
        return d0.functions.max(self, axis, keepdims)

    def to_cpu(self):
        if self.data is not None:
            self.data = d0.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = d0.cuda.as_cupy(self.data)


def as_array(x, array_module=np):
    if array_module.isscalar(x):
        return array_module.array(x)
    return x


def as_variable(x):
    if isinstance(x, Variable):
        return x
    return Variable(x)


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]

        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        if len(outputs) > 1:
            return outputs
        return outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        self.x_shapes = (x0.shape, x1.shape)
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy

        if self.x_shapes[0] != self.x_shapes[1]:
            gx0 = d0.functions.sum_to(gx0, self.x_shapes[0])
            gx1 = d0.functions.sum_to(gx1, self.x_shapes[1])

        return gx0, gx1


def add(x0, x1):
    x1 = as_array(x1, d0.cuda.get_array_module(x0.data))
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1, gy * x0


def mul(x0, x1):
    x1 = as_array(x1, d0.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1, d0.cuda.get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1, d0.cuda.get_array_module(x0.data))
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs

        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1, d0.cuda.get_array_module(x0.data))
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1, d0.cuda.get_array_module(x0.data))
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * self.c * x ** (self.c - 1)
        return gx


def pow(x, c):
    return Pow(c)(x)


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__getitem__ = d0.functions.get_item


class Parameter(Variable):
    pass
