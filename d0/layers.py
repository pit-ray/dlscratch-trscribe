import os
import weakref
import numpy as np

from d0.core import Parameter
from d0 import functions as F
from d0 import cuda
from d0.utils import pair


class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )

        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]

        if len(outputs) > 1:
            return outputs
        return outputs[0]

    def forward(self, *inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = getattr(self, name)

            if isinstance(obj, Layer):
                yield from obj.params()  # recursive generator
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, params_dict, parent_key=''):
        for name in self._params:
            obj = getattr(self, name)
            key = '{}/{}'.format(parent_key, name) if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)

        array_dict = {}
        for key, param in params_dict.items():
            if param is not None:
                array_dict[key] = param.data

        try:
            np.savez_compressed(path, **array_dict)

        except (Exception, KeyboardInterrupt):
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


class Linear(Layer):
    def __init__(
            self,
            out_size,
            nobias=False,
            in_size=None,
            dtype=np.float32):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')

        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        base = xp.random.randn(self.in_size, self.out_size).astype(self.dtype)
        self.W.data = xp.sqrt(1 / self.in_size) * base

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W(cuda.get_array_module(x))

        y = F.linear(x, self.W, self.b)
        return y


class Conv2d(Layer):
    def __init__(
            self,
            out_channels,
            kernel_size,
            stride=1,
            pad=0,
            nobias=False,
            dtype=np.float32,
            in_channels=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = self.kernel_size
        scale = np.sqrt(1 / C * KH * KW)
        self.W.data = scale * xp.random.randn(OC, C, KH, KW).astype(self.dtype)

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        return F.conv2d(x, self.W, self.b, self.stride, self.pad)


class RNN(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()

        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True)

        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))

        self.h = h_new
        return h_new


class LSTM(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()

        H, I = hidden_size, in_size

        self.x_forget = Linear(H, in_size=I)
        self.x_input = Linear(H, in_size=I)
        self.x_output = Linear(H, in_size=I)
        self.x_update = Linear(H, in_size=I)

        self.h_forget = Linear(H, in_size=I, nobias=True)
        self.h_input = Linear(H, in_size=I, nobias=True)
        self.h_output = Linear(H, in_size=I, nobias=True)
        self.h_update = Linear(H, in_size=I, nobias=True)

        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        if self.h is None:
            f = F.sigmoid(self.x_forget(x))
            i = F.sigmoid(self.x_input(x))
            o = F.sigmoid(self.x_output(x))
            u = F.tanh(self.x_update(x))

        else:
            f = F.sigmoid(self.x_forget(x) + self.h_forget(self.h))
            i = F.sigmoid(self.x_input(x) + self.h_input(self.h))
            o = F.sigmoid(self.x_output(x) + self.h_output(self.h))
            u = F.tanh(self.x_update(x) + self.h_update(self.h))

        if self.c is None:
            c_new = i * u
        else:
            c_new = f * self.c + i * u

        h_new = o * F.tanh(c_new)

        self.h = h_new
        self.c = c_new

        return h_new
