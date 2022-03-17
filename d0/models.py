import os

import numpy as np

from d0 import Layer
from d0 import utils

from d0 import functions as F
from d0 import layers as L


class Model(Layer):
    def plot(self, *inputs, filename=os.path.join('.dezero', 'model.png')):
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


class VGG16(Model):
    def __init__(self, pretrained=False):
        super().__init__()

        conv_layers = [
            [64, 64],
            [128, 128],
            [256, 256, 256],
            [512, 512, 512],
            [512, 512, 512]
        ]

        self.conv_nums = []

        for i, channels in enumerate(conv_layers):
            self.conv_nums.append(len(channels))
            for j, out_c in enumerate(channels):
                conv = L.Conv2d(out_c, kernel_size=3, stride=1, pad=1)
                setattr(self, 'conv{}_{}'.format(i + 1, j + 1), conv)

        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(1000)

        if pretrained:
            weights_path = utils.get_file(
                'https://github.com/koki0702/dezero-models/releases/download/'
                'v0.1/vgg16.npz')
            self.load_weights(weights_path)

    def forward(self, x):
        for i, conv_num in enumerate(self.conv_nums):
            for j in range(conv_num):
                conv = getattr(self, 'conv{}_{}'.format(i + 1, j + 1))
                x = F.relu(conv(x))

            x = F.max_pool2d(x, 2, 2)

        x = F.reshape(x, (x.shape[0], -1))
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x

    @staticmethod
    def preprocess(image, size=(224, 224), dtype=np.float32):
        image = image.convert('RGB')
        if size:
            image = image.resize(size)
        image = np.asarray(image, dtype=dtype)
        image = image[:, :, ::-1]
        image -= np.array([103.939, 116.779, 123.68], dtype=dtype)
        image = image.transpose((2, 0, 1))
        return image
