from numpy.oldnumeric.random_array import random_integers

__author__ = 'gavr'

import time
import PIL.Image
import StringIO

import numpy
from math import sqrt

import theano
import theano.tensor as T
import re
import os

from theano.tensor.shared_randomstreams import RandomStreams

class RBM:
    def __init__(self, hidden, visible, rnd, theanoRnd, W = None, hBias = None, vBias = None):
        self.hidden = hidden
        self.visible = visible
        self.rnd = rnd
        self.theanoRnd = theanoRnd

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
        sizeOfBlockForLearn = T.iscalar()
        Sample = T.vector()
        countGibbsSteps = T.iscalar()
        learningRate = T.fscalar()
        index = T.iscalar()

        # functions for computing probabilities
        computeProbabilitiesHByV_format = lambda sample: T.nnet.sigmoid(T.dot(sample, W) + hBias)
        computeProbabilitiesVByH_format = lambda sample: T.nnet.sigmoid(T.dot(W, sample) + vBias)
        # function for generate realization by probabilities
        sample_format = lambda probabilities: self.theanoRnd.binomial( \
                size=probabilities.shape, n=1, p=probabilities, dtype='floatX')
        # functions for sampling
        samplingHbyV_format = lambda sample: \
            sample_format(computeProbabilitiesHByV_format(sample))
        samplingVbyH_format = lambda sample: \
            sample_format(computeProbabilitiesVByH_format(sample))
        # function for make one gibbs-step
        gibbsOne_format = lambda sample: samplingVbyH_format(samplingHbyV_format(sample))
        # function for generate initial state for visible varibles
        generateRandomVisibles_format = self.theanoRnd.binomial(size=vBias.shape, n=1, p=T.ones_like(vBias) * 0.5, dtype='floatX')
        # template function for making gibbs
        template = lambda x: theano.scan(fn=gibbsOne_format, \
                                         outputs_info=x, \
                                         n_steps=countGibbsSteps)
        # function for gibbs from sample
        gibbs_format, updates = template(Sample)
        gibbs_format = gibbs_format[-1]
        # save this function
        self.gibbs = theano.function(inputs=[Sample, countGibbsSteps], outputs=gibbs_format, updates=updates)

        # function for gibbs from random generate
        gibbsFromRnd_format, updates_gibbs_rnd = template(generateRandomVisibles_format)
        gibbsFromRnd_format = gibbsFromRnd_format[-1]
        # save this function
        self.gibbsFromRnd = theano.function(inputs=[countGibbsSteps], outputs=gibbsFromRnd_format, updates=updates_gibbs_rnd)

        # function input vSample generate hSample and return [vSample.Transpose * hSample, vSample, hSample]
        maketriple_format = \
            lambda x: [T.outer(x, samplingHbyV_format(x)), x, samplingHbyV_format(x)]
        #func = theano.function(inputs=[countGibbsSteps], outputs=maketriple_format(gibbsFromRnd_format), updates=updates_gibbs_rnd)
        #print func(1)
        # fucntion for calcEModel
        emodelmat, emodelV, emodelH = maketriple_format(gibbsFromRnd_format)
        # Necessary function for calc Edata
        def eDataCalc(x, mat, vecV, vecH):
            m1, v1, h1 = maketriple_format(x)
            return [mat + m1, v1 + vecV, h1 + vecH]
        # create loop
        [edatamat, edataV, edataH], edata_updates = theano.scan(fn=eDataCalc, outputs_info=[T.zeros_like(W), T.zeros_like(vBias), T.zeros_like(hBias)], sequences=data)
        edatamat, edataV, edataH = edatamat[-1], edataV[-1], edataH[-1]
        # edata depends of sizeOfBlockForLearn, data
        # calc avg and deriviative
        sizeOfBlockForLearnFloat = T.cast(sizeOfBlockForLearn, dtype="floatX")
        egrad = [(-edatamat / sizeOfBlockForLearnFloat + emodelmat), \
                 (-edataV / sizeOfBlockForLearnFloat + emodelV), \
                 (-edataH / sizeOfBlockForLearnFloat + emodelH)]
        # egrad depedens of sizeOfBlockForLearn, data, learningRate, countGibbsSteps
        updates = edata_updates + updates_gibbs_rnd
        params = [W, vBias, hBias]
        for p, g in zip(params, egrad):
            updates[p] = p - g * learningRate
        # save function
        self.grad_step = theano.function([sizeOfBlockForLearn, learningRate, countGibbsSteps, data], egrad, updates=updates)

#        TODO: return not all gradient, only information about max, min, ExpectValue by W, hBias, vBias

    def saveTo(self, strIo):
        if isinstance(strIo, StringIO.StringIO):
            func = lambda theano_func: re.sub('array\(|\)|\n|\t|\[|\][^,]', '',repr(theano_func()))
            fget = lambda var: theano.function([], var)
            strIo.write(repr(self.visible) + "\n")
            strIo.write(repr(self.hidden) + "\n")
            strIo.write(func(fget(self.hBias)) + "\n")
            strIo.write(func(fget(self.vBias)) + "\n")
            strIo.write(func(fget(self.W)) + "\n")

def open(strio):
    array = strio.getvalue().split('\n')
    parse_vector = lambda str: map(float, str.split(','))
    parse_matrix = lambda str: [parse_vector(substr) for substr in str.split('],')]
    hBias = theano.shared(numpy.asarray(parse_vector(array[2])), borrow=True)
    vBias = theano.shared(numpy.asarray(parse_vector(array[3])), borrow=True)
    W = theano.shared(numpy.asarray(parse_matrix(array[4])), borrow=True)

    numpyRng = numpy.random.RandomState(1234)
    theanoRng = RandomStreams(numpyRng.randint(2 ** 30))
    return RBM(int(array[1]), int(array[0]), numpyRng, theanoRng, W, hBias, vBias)

def createSimpleRBM(hidden, visible):
    numpyRng = numpy.random.RandomState(1234)
    theanoRng = RandomStreams(numpyRng.randint(2 ** 30))
    return RBM(hidden, visible, numpyRng, theanoRng)

def convertImageToVector(image):
    return numpy.asarray(list(image.getdata()))

def convertImagesToVector(images):
    return [convertImageToVector(image) for image in images]

def convertVectorToImage(appearance, vector):
    im = appearance.copy()
    im.putdata(vector)
    return im

def Learn(rbm, data, countStep, learningRate, gibbsStep, func=None, zipManager=None):
    # TODO tic-toc time;
    # TODO work with ZipManager
    # TODO test ZipManager
    if func is None:
        for idx in range(0, countStep): rbm.grad_step(len(data), numpy.asarray(learningRate, dtype='float32'), gibbsStep, data);
    else:
        for idx in range(0, countStep): d1 = func(idx, data); rbm.grad_step(len(d1), numpy.asarray(learningRate, dtype='float32'), gibbsStep, d1);
    
#TODO make RBM.learn

'''
Example

'''

import PIL.Image
import PIL.ImageDraw
import PIL.ImagePalette

def generatorImage(size):
    image = PIL.Image.new(mode = "P", size = (size, size))
    image.putpalette([255, 255, 255, 0, 0, 0])
    draw = PIL.ImageDraw.Draw(image)
    f = lambda x, y: random_integers(y, minimum=x)
    draw.line((f(size / 2, size), f(size / 2, size), f(1, size / 2), f(1, size / 2)), fill = 1)
    return image

size = 5
# generate data
data = [convertImageToVector(generatorImage(size)) for i in range(0, 100)]
# create rbm
#   first param is count hidden
#   second param is count visible
rbm = createSimpleRBM(10, size * size)
res = []
res1 = []
for idx in range(0, 10):
    print idx
    print rbm.grad_step(len(data), numpy.asarray(0.01, dtype='float32'), 10, data)
   #  res += [convertVectorToImage(generatorImage(si), rbm.gibbsFromRnd(10))]
   # res1 += [convertVectorToImage(generatorImage(40), rbm.gibbsFromRnd(2))]
#for a in res:
 #   a.show()
    #b.show()

#convertVectorToImage(generatorImage(size), rbm.gibbs(convertImageToVector(generatorImage(size)), 40)).show()
#convertVectorToImage(generatorImage(size), rbm.gibbs(convertImageToVector(generatorImage(size)), 2)).show()
#convertVectorToImage(generatorImage(size), rbm.gibbs(data[10], 10)).show()
#convertVectorToImage(generatorImage(size), rbm.gibbs(data[10], 2)).show()

strio = StringIO.StringIO()
rbm.saveTo(strio)
print len(strio.getvalue().split('\n'))
#print strio.
rbm1 = open(strio)