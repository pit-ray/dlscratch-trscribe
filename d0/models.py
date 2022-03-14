from d0 import Layer
from d0 import utils

from d0 import functions as F
from d0 import layers as L


class Model(Layer):
    def plot(self, *inputs, filename='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, filename, verbose=True)


class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layer_num = len(fc_output_sizes)

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)

    def forward(self, x):
        for i in range(self.layer_num - 1):
            x = self.activation(getattr(self, 'l' + str(i))(x))
        return getattr(self, 'l' + str(self.layer_num - 1))(x)
