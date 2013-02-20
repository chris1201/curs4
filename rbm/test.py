__author__ = 'gavr'
import time
import PIL.Image
import PIL.ImageDraw
import PIL.ImagePalette
# save
import StringIO

import numpy
from math import sqrt
import theano
import theano.tensor as T
from numpy.oldnumeric.random_array import random_integers
from theano.tensor.shared_randomstreams import RandomStreams

import re
import os
import sys


q = T.tensor3()

print q
z = numpy.asarray([[[1, 2, -1], [3, 4, -1]], [[5, 6, -1], [7, 8, -1]]])
print z
q.dimshuffle(0, 2, 1)
#f1, upd = theano.scan(lambda a: T.dot(a, a.T), sequences=q)
#f2 = T.sum(f1, axis=(1, 2))
#f2 = T.addbroadcast(f2)
#f3 = theano.function([q], [f2, q[0], (f2 + q[0])])
a, b, c = q.shape
eee = T.zeros((a, b))
f4 = theano.function([q], eee)
#print f3(z)
print f4(z)