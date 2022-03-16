from d0.im2col import im2col
from d0.utils import pair, get_conv_outsize
from d0.functions import linear
from d0 import as_variable


def conv2d_simple(x, weight, b=None, stride=1, pad=0):
    x, weight = as_variable(x), as_variable(weight)

    N, C, H, W = x.shape
    OC, C, KH, KW = weight.shape
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    col = im2col(x, (KH, KW), (SH, SW), (PH, PW), to_matrix=True)

    weight = weight.reshape(OC, -1).transpose()
    t = linear(col, weight, b)
    y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
    return y


def conv2d(x, weight, b=None, stride=1, pad=0):
    return conv2d_simple(x, weight, b=b, stride=stride, pad=pad)


def max_pool2d_simple(x, kernel_size, stride=1, pad=0):
    x = as_variable(x)

    N, C, H, W = x.shape

    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    col = im2col(x, kernel_size, stride, pad, to_matrix=True)
    col = col.reshape(-1, KH * KW)
    y = col.max(axis=1)
    y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
    return y


def max_pool2d(x, kernel_size, stride=1, pad=0):
    return max_pool2d_simple(x, kernel_size, stride=stride, pad=pad)
