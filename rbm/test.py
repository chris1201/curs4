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


a = theano.shared(1)
values,updates = theano.scan( lambda : [a, {a:a+1}], n_steps = 10 )
print updates
print values
b = a+1
c = updates[a] + 1
f = theano.function([], [values, b,c], updates = updates)

print b
print c
print a.value
print f()