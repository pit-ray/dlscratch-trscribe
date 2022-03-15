import numpy as np


class Compose:
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, x):
        if not self.transforms:
            return x

        for t in self.transforms:
            x = t(x)
        return x


class Flatten:
    def __call__(self, x):
        return x.flatten()


class AsType:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype)


class Normalize:
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x - self.mean) / self.std
