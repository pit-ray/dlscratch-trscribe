import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image

from d0.datasets import Dataset
from d0.models import VGG16
from d0 import utils, test_mode


def test1():
    model = VGG16(pretrained=True)

    x = np.random.randn(1, 3, 224, 224).astype(np.float32)
    model.plot(x)


class ImageNet(Dataset):

    def __init__(self):
        NotImplemented

    @staticmethod
    def labels():
        url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
        path = utils.get_file(url)
        with open(path, 'r') as f:
            labels = eval(f.read())
        return labels


def predict(model, img_path):
    img = Image.open(img_path)
    x = VGG16.preprocess(img)
    x = x[None, :, :, :]

    with test_mode():
        y = model(x)
    predict_id = np.argmax(y.data)

    labels = ImageNet.labels()
    print(img_path, labels[predict_id])


def test2():
    img_paths = []

    url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg'
    img_paths.append(utils.get_file(url))

    img_paths.append(os.path.join('.dezero', 'test1.jpg'))
    img_paths.append(os.path.join('.dezero', 'test2.jpg'))
    img_paths.append(os.path.join('.dezero', 'test3.jpg'))
    img_paths.append(os.path.join('.dezero', 'test4.jpg'))
    img_paths.append(os.path.join('.dezero', 'test5.jpg'))

    model = VGG16(pretrained=True)

    for img_path in img_paths:
        predict(model, img_path)


test2()
