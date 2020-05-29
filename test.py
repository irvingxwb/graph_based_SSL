import tensorflow as tf
from helper import *
import logging
from scipy.sparse import csr_matrix
import numpy as np
from sys import getsizeof

size = 10000

a = np.zeros([size, size])
b = csr_matrix((size, size))
b[0, 0:10] = 1

c = [1,5,7,3,8,9,2]
print(sorted(c)[-3:])