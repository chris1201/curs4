__author__ = 'gavr'
import time
import PIL.Image
import PIL.ImageDraw
import PIL.ImagePalette
import StringIO

from theano.gof.python25 import OrderedDict
import theano.updates
import numpy
from math import sqrt
import theano
import theano.tensor as T
from numpy.oldnumeric.random_array import random_integers
from theano.tensor.shared_randomstreams import RandomStreams

class RTRBM:
    def __init__(self, hidden, visible, rnd, theanoRnd, W = None, hBias = None, vBias = None):
        self.hidden = hidden
        self.visible = visible

        # initial values
        if W is None:
            WInit = numpy.asarray(rnd.uniform(
                      low=-4 * sqrt(6. / (hidden + visible)),
                      high=4 * sqrt(6. / (hidden + visible)),
                      size=(visible, hidden)),
                      dtype=theano.config.floatX)
            W = theano.shared(WInit, borrow = False)

        if hBias is None:
            hBiasInit = numpy.zeros(hidden)

            hBias = theano.shared(hBiasInit, borrow = True)

        if vBias is None:
            vBiasInit = numpy.zeros(visible)

            vBias = theano.shared(vBiasInit, borrow = True)

        # save in class
        self.W = W
        self.vBias = vBias
        self.hBias = hBias
        # varibles
        data = T.matrix()
        Sample = T.vector()
        countGibbsSteps = T.iscalar()
        learningRate = T.fscalar()
        # functions for computing probabilities
        computeProbabilitiesHByV_format = lambda sample: T.nnet.sigmoid(T.dot(sample, W) + hBias)
        computeProbabilitiesVByH_format = lambda sample: T.nnet.sigmoid(T.dot(sample, W.T) + vBias)
        # function for generate realization by probabilities
        sample_format = lambda probabilities: theanoRnd.binomial( \
                size=probabilities.shape, n=1, p=probabilities, dtype='floatX')
        # functions for sampling
        samplingHbyV_format = lambda sample: \
            sample_format(computeProbabilitiesHByV_format(sample))
        samplingVbyH_format = lambda sample: \
            sample_format(computeProbabilitiesVByH_format(sample))
        # function for make one gibbs-step
        gibbsOne_format = lambda sample: samplingVbyH_format(samplingHbyV_format(sample))
        # function for generate initial state for visible varibles
        generateRandomVisibles_format = theanoRnd.binomial(size=vBias.shape, n=1, p=T.ones_like(vBias) * 0.5, dtype='floatX')
        self.generateVisibles = theano.function([], generateRandomVisibles_format)
        # template function for making gibbs
        template = lambda x: theano.scan(fn=gibbsOne_format, \
                                         outputs_info=x, \
                                         n_steps=countGibbsSteps)
        gu = theano.updates.OrderedUpdates()
        def func(x):
            res, updates = template(x)
            res = res[-1]
            return [res, {k: v for k, v in OrderedDict(updates).iteritems()}]

        def func1(X):
            q, upd = theano.scan(func, sequences=X)
            f = theano.function([countGibbsSteps, X], q, updates=upd)
            print upd
            print q

        func1(data)

def createSimpleRBM(hidden, visible):
    numpyRng = numpy.random.RandomState(1234)
    theanoRng = RandomStreams(numpyRng.randint(2 ** 30))
    return RTRBM(hidden, visible, numpyRng, theanoRng)

q = createSimpleRBM(10, 12)
