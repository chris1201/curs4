from re import template
import re

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
    def __init__(self, hidden, visible, rnd, theanoRnd, W = None, hBias = None, vBias = None, W1 = None, W2 = None, h_lid_0 = None, not_random = None):
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

        if W1 is None:
            W1Init = numpy.asarray(rnd.uniform(
                      low=-4 * sqrt(6. / (hidden + visible)),
                      high=4 * sqrt(6. / (hidden + visible)),
                      size=(visible, hidden)),
                      dtype=theano.config.floatX)
            W1 = theano.shared(W1Init, borrow = False)

        if W2 is None:
            W2Init = numpy.asarray(rnd.uniform(
                      low=-4 * sqrt(6. / (hidden + hidden)),
                      high=4 * sqrt(6. / (hidden + hidden)),
                      size=(hidden, hidden)),
                      dtype=theano.config.floatX)
            W2 = theano.shared(W2Init, borrow = False)


        if hBias is None:
            hBiasInit = numpy.zeros(hidden, dtype=theano.config.floatX)

            hBias = theano.shared(hBiasInit, borrow = True)

        if vBias is None:
            vBiasInit = numpy.zeros(visible)
            vBias = theano.shared(vBiasInit, borrow = True)

        if h_lid_0 is None:
            h_lid_0_Init = numpy.zeros(hidden)
            h_lid_0 = theano.shared(h_lid_0_Init, borrow = True)

        # save in class
        self.W = W
        self.W1 = W1
        self.W2 = W2
        self.vBias = vBias
        self.hBias = hBias
        self.h_lid_0 = h_lid_0
        # varibles
        data = T.tensor3()
        Sample = T.matrix()
        countGibbsSteps = T.iscalar()
        lengthOfSequence = T.iscalar()
        learningRate = T.fscalar()
        # functions for computing probabilities
        computeProbabilitiesHByV_format = lambda sample, hBiases: T.nnet.sigmoid(T.dot(sample, W) + hBiases)
        computeProbabilitiesVByH_format = lambda sample, vBiases: T.nnet.sigmoid(T.dot(sample, W.T) + vBiases)
        # function for generate realization by probabilities
        sample_format = lambda probabilities: theanoRnd.binomial( \
                size=probabilities.shape, n=1, p=probabilities, dtype='floatX')
        # functions for sampling
        samplingHbyV_format = lambda sample, hBiases: \
            sample_format(computeProbabilitiesHByV_format(sample, hBiases))
        samplingVbyH_format = lambda sample, vBiases: \
            sample_format(computeProbabilitiesVByH_format(sample, vBiases))
        # function for make one gibbs-step
        gibbsOne_format = lambda sample, vBiases, hBiases: samplingVbyH_format(samplingHbyV_format(sample, hBiases), vBiases)
        gibbsOneWithOutCoin_format = lambda sample, vBiases, hBiases: computeProbabilitiesVByH_format(computeProbabilitiesHByV_format(sample, hBiases), vBiases)
        # function for generate initial state for visible varibles
        #   Parameter: lengthOfSequence
        generateRandomVisiblesForOneTime_format = lambda: theanoRnd.binomial(size=vBias.shape, n=1, p=T.ones_like(vBias) * 0.5, dtype='floatX')
        def generateRandomVisibles_format(length):
            rnd, unusedUpdates = theano.scan(generateRandomVisiblesForOneTime_format, n_steps=length)
            return rnd
        # templates of Varibles for calculate h_lid by previous value
        calc_h_lid = lambda h_lid_old, sample: T.nnet.sigmoid(T.dot(sample, W) + T.dot(W2, h_lid_old) + hBias)
        calc_hBiases = lambda h_lid: hBias + T.dot(h_lid, W2.T)
        calc_vBiases = lambda h_lid: vBias + T.dot(h_lid, W1.T)
        # Make For One input Image(sample) gibbs-sampling for one-time
        #   Parameter: countGibbsStep
        def gibbsSamplingForOneStepTime(sample, h_lid):
            res, updates = theano.scan(gibbsOne_format, outputs_info=sample, non_sequences=[calc_vBiases(h_lid), calc_hBiases(h_lid)], n_steps=countGibbsSteps)
            res = res[-1]
            return [[res, calc_h_lid(h_lid, res)], updates]
        # Make For input image gibbs-sampling for one-time
        def gibbsSamplingForOneStepTimeWithOutCoin(sample, h_lid):
            res, updates = theano.scan(gibbsOneWithOutCoin_format, outputs_info=sample, non_sequences=[calc_vBiases(h_lid), calc_hBiases(h_lid)], n_steps=countGibbsSteps)
            res = res[-1]
            return [[res, calc_h_lid(h_lid, res)], updates]
        # Make gibbs-sampling for all-time
        #   Parameter: countGibbsStep
        def gibbsSamplingForAllTime(sample, start_h_lid):
            [samp_res, hlids], updates = theano.scan(gibbsSamplingForOneStepTime, sequences=sample, outputs_info=[None, start_h_lid])
            return samp_res, hlids, updates

        def gibbsSamplingForAllTimeWithOutCoin(sample, start_h_lid):
            [samp_res, hlids], updates = theano.scan(gibbsSamplingForOneStepTimeWithOutCoin, sequences=sample, outputs_info=[None, start_h_lid])
            return samp_res, hlids, updates

        # usual gibbs-sampling
        res, _, upds = gibbsSamplingForAllTime(Sample, h_lid_0)
        self.gibbsSampling = theano.function([Sample, countGibbsSteps], res, updates=upds)
        # usual random gibbs-sampling
        res = generateRandomVisibles_format(lengthOfSequence)
        res, _, upds = gibbsSamplingForAllTime(res, h_lid_0)
        self.gibbsSamplingFromRnd = theano.function([lengthOfSequence, countGibbsSteps], res, updates=upds)

        # sampling without coin
        res, _, upds = gibbsSamplingForAllTimeWithOutCoin(Sample, h_lid_0)
        self.gibbsSamplingWithOutCoin = theano.function([Sample, countGibbsSteps], res, updates=upds)

        # prediction
        res0, h_lids, upds0 = gibbsSamplingForAllTime(Sample, h_lid_0)
        randomData = generateRandomVisibles_format(lengthOfSequence)
        res1, _, upds1 = gibbsSamplingForAllTime(randomData, h_lids[-1])
        res = T.concatenate([res0, res1]);
        self.gibbsSamplingPrediction = theano.function([Sample, lengthOfSequence, countGibbsSteps], res, updates=upds0 + upds1)

        # prediction without coin
        res0, h_lids, upds0 = gibbsSamplingForAllTimeWithOutCoin(Sample, h_lid_0)
        randomData = generateRandomVisibles_format(lengthOfSequence)
        res1, _, upds1 = gibbsSamplingForAllTimeWithOutCoin(randomData, h_lids[-1])
        res = T.concatenate([res0, res1]);
        self.gibbsSamplingPredictionWithOutCoin = theano.function([Sample, lengthOfSequence, countGibbsSteps], res, updates=upds0 + upds1)

        def calc_Energy(sample, vBiases, hBiases):
            q = hBiases + T.dot(sample, W)
            # energy for One Time and for all objects in sample
            energyOne = T.dot(sample, vBiases.T) + T.sum(T.log(1 + T.exp(q)))
            energy = -T.sum(energyOne)
            return energy, vBiases, hBiases
        
        # Calc h_lids
        def calc_h_lids(sample, h_lid_start):
            h_lids, updates = theano.scan(lambda u, v: calc_h_lid(v, u), sequences=sample[0:-1], outputs_info=h_lid_start)
            a, b = T.shape(h_lids)
            shape = (a + 1, b)
            h_lids = T.flatten(h_lids)
            h_lids = T.concatenate([h_lid_start, h_lids])
            h_lids = T.reshape(h_lids, shape)
            return h_lids, updates

        def calc_grad_Energy_For_One_Object(sample, h_lids, evaluatesample, updates):
            vBiases = calc_vBiases(h_lids)
            hBiases = calc_hBiases(h_lids)
            Pdata, _, _ = calc_Energy(sample, vBiases, hBiases)
            Pmodel, _, _ = calc_Energy(evaluatesample, vBiases, hBiases)
            P = Pdata - Pmodel
            grad = theano.grad(P, [W, vBiases, hBiases], consider_constant=[sample, evaluatesample, W1, W2, hBias, vBias])
            gradUByW1 = T.dot(grad[1].T, h_lids);
            gradUByW2 = T.dot(grad[2].T, h_lids);
            gradUByhBias = T.sum(grad[2], axis=0);
            gradUByvBias = T.sum(grad[1], axis=0);
            gradUByW = grad[0];
            gradHLid0 = grad[2][0];
            #block = [W1, W2, hBias, vBias, W, hlid0]
            grad_block_return = [P, gradUByW1, gradUByW2, gradUByhBias, gradUByvBias, gradUByW, gradHLid0]
            return grad_block_return, updates

        def calc_grad_Energy_For_One_Object_By_Data(sample):
            new_sample, _, updates = gibbsSamplingForAllTime(sample, h_lid_0)
            h_lids, updates_h_lid = calc_h_lids(sample, h_lid_0)
            return calc_grad_Energy_For_One_Object(sample, h_lids, new_sample, updates+updates_h_lid)

        def calc_grad_Energy_For_One_Object_By_Rnd(sample):
            new_sample, _, updates = gibbsSamplingForAllTime(generateRandomVisibles_format(sample.shape[0]), h_lid_0)
            h_lids, updates_h_lid = calc_h_lids(sample, h_lid_0)
            return calc_grad_Energy_For_One_Object(sample, h_lids, new_sample, updates+updates_h_lid)

        def calc_grad_Energy_For_Input_Objects(samples, func):
            Q, updates = theano.scan(func, sequences=samples)
            meanQ = [T.mean(grad, axis=0) for grad in Q]
            return meanQ, updates

        #block = [W1, W2, hBias, vBias, W, h_lid_0]
        #grad_block_return = [gradUByW1, gradUByW2, gradUByhBias, gradUByvBias, W]
        block = [W1, W2, hBias, vBias, W, h_lid_0]
        if not_random is None:
            func = calc_grad_Energy_For_One_Object_By_Rnd
        else:
            func = calc_grad_Energy_For_One_Object_By_Data
        alls, upds = calc_grad_Energy_For_Input_Objects(data, func)
        for u, v in zip(block, alls[1:]):
            upds[u] = u - learningRate * v
        self.grad_step = theano.function([data, countGibbsSteps, learningRate], alls[0], updates=upds)
        self.step = theano.function([], block)

    def save(self):
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
        strIo.write(func(convertingMatrix(fget(self.W1)())) + "\n")
        strIo.write(func(convertingMatrix(fget(self.W2)())) + "\n")
        strIo.write(func(convertingVector(fget(self.h_lid_0)())) + "\n")
        return strIo
        # save order^ visible, hidden, hBias, vBias, W, W1, W2


# TODO save RTRBM
# TODO test RTRBM
# TODO Apply RTRBM for clocks.

def createSimpleRTRBM(hidden, visible):
    numpyRng = numpy.random.RandomState(1234)
    theanoRng = RandomStreams(numpyRng.randint(2 ** 30))
    return RTRBM(hidden, visible, numpyRng, theanoRng)

def openRTRBM(strio):
    print strio
    array = strio.split('\n')
    parse_vector = lambda str: map(float, str.split(','))
    parse_matrix = lambda str: [parse_vector(substr) for substr in str.split('],')]
    # open order^ visible, hidden, hBias, vBias, W, W1, W2, h_lid0
    #               0       1       2       3    4  5   6   7
    hBias = theano.shared(numpy.asarray(parse_vector(array[2])), borrow=True)
    vBias = theano.shared(numpy.asarray(parse_vector(array[3])), borrow=True)
    W = theano.shared(numpy.asarray(parse_matrix(array[4])), borrow=True)
    W1 = theano.shared(numpy.asarray(parse_matrix(array[5])), borrow=True)
    W2 = theano.shared(numpy.asarray(parse_matrix(array[6])), borrow=True)
    h_lid_0 = theano.shared(numpy.asarray(parse_vector(array[7])))

    numpyRng = numpy.random.RandomState(1234)
    theanoRng = RandomStreams(numpyRng.randint(2 ** 30))
    return RTRBM(int(array[1]), int(array[0]), numpyRng, theanoRng, W, hBias, vBias, W1, W2, h_lid_0=h_lid_0)


