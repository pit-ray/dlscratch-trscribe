import math
import numpy as np
from d0 import cuda


class DataLoader:
    def __init__(
            self,
            dataset,
            batch_size,
            shuffle=True,
            gpu=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.gpu = gpu

        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(self.data_size)
        else:
            self.index = np.arange(self.data_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration()

        i, N = self.iteration, self.batch_size

        batch_index = self.index[i * N:(i + 1) * N]

        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1

        return x, t

    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True


class SeqDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, gpu=False):
        super().__init__(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                gpu=gpu)

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration()

        jump = self.data_size // self.batch_size

        i, N = self.iteration, self.batch_size

        # b * jump : start position of each batch
        # If the batch like this.
        # |#####, #####, ###|
        #  01234  56789  abc
        #
        # It returns mod-based index.
        # |#####, #####, #####|
        #  01234  56789  abc01
        batch_index = [(b * jump + i) % self.data_size for b in range(N)]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1

        return x, t
