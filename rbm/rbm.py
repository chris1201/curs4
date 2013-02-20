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
        cost = T.mean(freeEnergy_format(data)) - (freeEnergy_format_vector(gibbsFromRnd_format_matrix))
        gradBlock = [W, vBias, hBias]
        gradient = theano.grad(cost, gradBlock, consider_constant=[data, gibbsFromRnd_format_matrix])
        updates = updates_gibbs_rnd
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

def convertImageToVector(image):
    return numpy.asarray(list(image.getdata()))

def convertVectorToImage(appearance, vector):
    im = appearance.copy()
    im.putdata(vector)
    return im

# save Data
def saveData(strio):
    file = open('data.txt', 'w')
    file.write(strio)
    file.close()

# readData from data.txt
def getStringData():
    file = open('data.txt', 'r')
    s = StringIO.StringIO()
    output = file.readlines()
    s.writelines(output)
    file.close()
    return s.getvalue()

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

"""

    Example
    
"""


def generatorImage(size):
    image = PIL.Image.new(mode = "P", size = (size, size))
    image.putpalette([255, 255, 255, 0, 0, 0])
    draw = PIL.ImageDraw.Draw(image)
    f = lambda x, y: random_integers(y, minimum=x)
    draw.line((f(1, size/2), f(1, size/2), f(size/2, size), f(size/2, size)), fill = 1)
    return image

def generatorWrongImage(size):
    image = PIL.Image.new(mode = "P", size = (size, size))
    image.putpalette([255, 255, 255, 0, 0, 0])
    draw = PIL.ImageDraw.Draw(image)
    f = lambda x, y: random_integers(y, minimum=x)
    draw.line((f(size / 2, size), f(1, size / 2), f(1, size / 2), f(size / 2, size)), fill = 1)
    return image

size = 20
# generate Data
datasize = 100
data = [convertImageToVector(generatorImage(size)) for i in range(0, datasize)]
rbm = createSimpleRBM(100, size * size)
#saveData(rbm.saveTo().getvalue())
#rbm = openRBM(getStringData())
print 'start train'

for idx in range(0, 200):
   # for inneridx in range(0, datasize):
    print idx, rbm.grad_step(data, numpy.asarray(0.01, dtype='float32'), 3)

print 'control train data'

for obj in data:
    print rbm.freeEnergy(obj)

print 'control train data'

data = [convertImageToVector(generatorImage(size)) for i in range(0, 10)]

for obj in data:
    print rbm.freeEnergy(obj)

print 'randomInfo'

for idx in range(0, 5):
    x = rbm.generateVisibles()
    print rbm.freeEnergy(x)
    x1 = rbm.gibbs(x, 1)
    print rbm.freeEnergy(x1)
    x2 = rbm.gibbs(x, 10)
    print rbm.freeEnergy(x2)

print 'WringImage'

for idx in range(0, 5):
    x = generatorWrongImage(size)
    x = convertImageToVector(x)
    print rbm.freeEnergy(x)
    x1 = rbm.gibbs(x, 1)
    print rbm.freeEnergy(x1)
    x2 = rbm.gibbs(x, 10)
    print rbm.freeEnergy(x2)

generatorImage(size).show()
convertVectorToImage(generatorImage(size), rbm.gibbs(convertImageToVector(generatorImage(size)), 1)).show()
convertVectorToImage(generatorImage(size), rbm.gibbs(convertImageToVector(generatorImage(size)), 2)).show()
convertVectorToImage(generatorImage(size), rbm.gibbs(convertImageToVector(generatorImage(size)), 5)).show()
convertVectorToImage(generatorImage(size), rbm.gibbs(convertImageToVector(generatorImage(size)), 10)).show()
convertVectorToImage(generatorImage(size), rbm.gibbs(convertImageToVector(generatorImage(size)), 20)).show()

convertVectorToImage(generatorImage(size), rbm.gibbsFromRnd(2)).show()
convertVectorToImage(generatorImage(size), rbm.gibbsFromRnd(5)).show()
convertVectorToImage(generatorImage(size), rbm.gibbsFromRnd(10)).show()
convertVectorToImage(generatorImage(size), rbm.gibbsFromRnd(20)).show()


saveData(rbm.saveTo().getvalue())
print 'saving has been made'