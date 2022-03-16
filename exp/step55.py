import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import d0


H, W = 4, 4
KH, KW = 3, 3
SH, SW = 1, 1
PH, PW = 1, 1

OH = d0.utils.get_conv_outsize(H, KH, SH, PH)
OW = d0.utils.get_conv_outsize(W, KW, SW, PW)
print(OH, OW)
