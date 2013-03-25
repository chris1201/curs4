__author__ = 'gavr'

import utils
import StringIO

import numpy
from math import sqrt
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import re

"""
freeEnergy(visibleSample) - calculate freeEnergy for sample of visible energy

generateVisibles() - generate visible varible from Bi(0.5, 1)

gibbs(visibleSample, count) - make gibbs-count for visibleSample

grad_step(data, learningRate, countGibbsStep) - make one gradient-step for training block-data,
    with rate-learningRate, and approximate E_model with countGibbsStep

gibbsFromRnd(count) - make gibbs-count for random generate visibles

saveTo() - create stringIO, where put all necessary data
"""
class RBM:
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
        # function for calc FreeEnergy
        def freeEnergy_format(sample):
            xdotw_plus_bias = T.dot(sample, W) + hBias
            xdotvbias = T.dot(sample, vBias)
            sum_log = T.sum(T.log(1 + T.exp(xdotw_plus_bias)), axis=1)
            return -sum_log - xdotvbias
        def freeEnergy_format_vector(sample):
            xdotw_plus_bias = T.dot(sample, W) + hBias
            xdotvbias = T.dot(sample, vBias)
            sum_log = T.sum(T.log(1 + T.exp(xdotw_plus_bias)), axis=0)
            return -sum_log - xdotvbias
        self.freeEnergy = theano.function([Sample], freeEnergy_format_vector(Sample))
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
        # function for gibbs from sample
        gibbs_format, updates = template(Sample)
        gibbs_format = gibbs_format[-1]
        # save this function
        self.gibbs = theano.function(inputs=[Sample, countGibbsSteps], outputs=gibbs_format, updates=updates)
        # make gibbs step for all samples in data
        # gibbs_matrix_format depends from data, countGibbsSteps
        gibbs_matrix_format, updates_gibbs_matrix = template(data)
        gibbs_matrix_format = gibbs_matrix_format[-1]
        # make gibbs-step from random
        gibbsFromRnd_format, updates_gibbs_rnd = template(generateRandomVisibles_format)
        gibbsFromRnd_format_matrix = gibbsFromRnd_format[-1]
        gibbsFromRnd_format = gibbsFromRnd_format[-1]
        # save this function
        self.gibbsFromRnd = theano.function(inputs=[countGibbsSteps], outputs=gibbsFromRnd_format, updates=updates_gibbs_rnd)
        # cost depends from data, countGibbsSteps
        cost = T.mean(freeEnergy_format(data)) - T.mean(freeEnergy_format(gibbs_matrix_format))
        gradBlock = [W, vBias, hBias]
        gradient = theano.grad(cost, gradBlock, consider_constant=[data, gibbs_matrix_format])
        updates = updates_gibbs_matrix
        for value, grad in zip(gradBlock, gradient):
            updates[value] = value - learningRate * grad
        self.grad_step = theano.function([data, learningRate, countGibbsSteps], cost, updates=updates)

    def saveTo(self):
        strIo = StringIO.StringIO()
        convertingVector = lambda x: '[ '+', '.join(map(str, x)) + '] '
        convertingMatrix = lambda y: '[' + '], '.join(map(convertingVector, y)) + '] '
        func = lambda str: re.sub('array\(|\)|\n|\t|\[|\][^,]', '', str)
        fget = lambda var: theano.function([], var)
        strIo.write(repr(self.visible) + "\n")
        strIo.write(repr(self.hidden) + "\n")
        strIo.write(func(convertingVector(fget(self.hBias)())) + "\n")
        strIo.write(func(convertingVector(fget(self.vBias)())) + "\n")
        strIo.write(func(convertingMatrix(fget(self.W)())) + "\n")
        return strIo

def createSimpleRBM(hidden, visible):
    numpyRng = numpy.random.RandomState(1234)
    theanoRng = RandomStreams(numpyRng.randint(2 ** 30))
    return RBM(hidden, visible, numpyRng, theanoRng)

# create RBM from string-text
def openRBM(strio):
    print strio
    array = strio.split('\n')
    parse_vector = lambda str: map(float, str.split(','))
    parse_matrix = lambda str: [parse_vector(substr) for substr in str.split('],')]
    hBias = theano.shared(numpy.asarray(parse_vector(array[2])), borrow=True)
    vBias = theano.shared(numpy.asarray(parse_vector(array[3])), borrow=True)
    W = theano.shared(numpy.asarray(parse_matrix(array[4])), borrow=True)

    numpyRng = numpy.random.RandomState(1234)
    theanoRng = RandomStreams(numpyRng.randint(2 ** 30))
    return RBM(int(array[1]), int(array[0]), numpyRng, theanoRng, W, hBias, vBias)


