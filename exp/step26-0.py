import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from d0 import Variable
from d0.utils import \
        _dot_var, \
        _dot_func
        # get_dot_graph


x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
y = x0 + x1

x0.name = 'x0'
x1.name = 'x1'
y.name = 'y'

print(_dot_var(x0))
print(_dot_var(x0, verbose=True))

print(_dot_func(y.creator))

# txt = get_dot_graph(y, verbose=False)
# print(txt)

#with open('sample.dot', 'w') as f:
#    f.write(txt)
