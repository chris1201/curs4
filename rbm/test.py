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

from PIL import Image

im = Image.new(mode='F', size = (40, 40))
data = im.getdata()
data = [value for value in data]

data[800] = 0.1
data[900] = 0.5
data[1000] = 1

data = map(lambda x: x * 256, data);
im.putdata(data)
print im.size
print im.mode
im.show()