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

import rbm
