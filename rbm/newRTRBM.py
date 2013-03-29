__author__ = 'gavr'

from theano.gof.python25 import OrderedDict
import theano.updates
import numpy
from math import sqrt
import theano
import theano.tensor as T
import StringIO
import re
from numpy.oldnumeric.random_array import random_integers
from theano.tensor.shared_randomstreams import RandomStreams
from tictoc import tic, toc
from theano.tensor.basic import TensorVariable

MODE_WITHOUT_COIN = 0
MODE_WITH_COIN = 1
MODE_WITH_COIN_EXCEPT_LAST = 2
MODE_WITHOUT_COIN_EXCEPT_LAST = 3

MODE_NAMES = ['W', 'WO', 'WELast', 'WOELast']

class BM:
    def __init__(self, theanoRnd):
        self.theanoRnd = theanoRnd
        self.list_function_for_gibbs = [self.computeProbabilityVByHByV, self.samplingVByHByV];
        
    def computeProbabilityHByV(self, sample, W, hBias):
        return T.nnet.sigmoid(T.dot(sample, W) + hBias)

    def computeProbabilityVByH(self, sample, W, vBias):
        return T.nnet.sigmoid(T.dot(sample, W.T) + vBias)

    def computeProbabilityVByHByV(self, sample, W, vBias, hBias):
        return self.computeProbabilityVByH(self.computeProbabilityHByV(sample, W, hBias), W, vBias)

    def generateRandomsFromBinoZeroOne(self, probability, num = 1):
        return self.theanoRnd.binomial( \
                size=probability.shape, n=num, p=probability, dtype='floatX')

    def samplingVByHByV(self, sample, W, vBias, hBias):
        return self.generateRandomsFromBinoZeroOne(self.computeProbabilityVByH(\
                self.generateRandomsFromBinoZeroOne(self.computeProbabilityHByV(sample, W, hBias)), W, vBias))

    def gibbs(self, sample, W, vBias, hBias, countSteps, function_mode):
        format, updates = self.gibbs_all(sample, W, vBias, hBias, countSteps, function_mode)
        return format[-1], updates

    def gibbs_all(self, sample, W, vBias, hBias, countSteps, function_mode):
        if function_mode < 2:
            gibbsOne_format = lambda sample: self.list_function_for_gibbs[function_mode](sample, W, vBias, hBias);
            format, updates = theano.scan(fn=gibbsOne_format, \
                                          outputs_info=sample, \
                                          n_steps=countSteps)
            return format, updates
        else:
            if function_mode == MODE_WITH_COIN_EXCEPT_LAST:
                gibbsOne_format = lambda sample: self.list_function_for_gibbs[MODE_WITH_COIN](sample, W, vBias, hBias);
                format, updates = theano.scan(fn=gibbsOne_format, \
                                          outputs_info=sample, \
                                          n_steps=countSteps - 1)
                gibbsOne_format = lambda sample: self.list_function_for_gibbs[MODE_WITHOUT_COIN](sample, W, vBias, hBias);
                res = gibbsOne_format(format[-1])
                res = T.concatenate([format, [res]])
                return res, updates
            else:
                gibbsOne_format = lambda sample: self.list_function_for_gibbs[MODE_WITHOUT_COIN](sample, W, vBias, hBias);
                format, updates = theano.scan(fn=gibbsOne_format, \
                                              outputs_info=sample, \
                                              n_steps=countSteps - 1)
                gibbsOne_format = lambda sample: self.list_function_for_gibbs[MODE_WITH_COIN](sample, W, vBias, hBias);
                res = gibbsOne_format(format[-1])
                res = T.concatenate([format, [res]])
                return res, updates


    def freeEnergy(self, sample, W, vBias, hBias):
        xdotw_plus_bias = T.dot(sample, W) + hBias
        xdotvbias = T.dot(sample, vBias)
        log = T.log(1 + T.exp(xdotw_plus_bias))
        # TODO condition about test is it matrix or is it vector
        if (len(sample.broadcastable) == 2):
            sum_log = T.sum(log, axis=1)
        else:
            sum_log = T.sum(log, axis=0)
        return -sum_log - xdotvbias

    def addGradientToUpdate(self, update, gradVaribles, grad, learningRate):
        for u, v in zip(gradVaribles, grad):
            update[u] = u - learningRate * v

    def saveTo(self, v, h, variblesVectors, variblesMatrices):
        strIo = StringIO.StringIO()
        convertingVector = lambda x: '[ '+', '.join(map(str, x)) + '] '
        convertingMatrix = lambda y: '[' + '], '.join(map(convertingVector, y)) + '] '
        func = lambda str: re.sub('array\(|\)|\n|\t|\[|\][^,]', '', str)
        fget = lambda var: theano.function([], var)()
        res_func = lambda var, input_func: func(input_func(fget(var))) + "\n"
        strIo.write(repr(v) + "\n")
        strIo.write(repr(h) + "\n")
        for x in variblesVectors:
            strIo.write(res_func(x, convertingVector))
        for x in variblesMatrices:
            strIo.write(res_func(x, convertingMatrix))
        return strIo.getvalue()

    def findMaxEnergy(self, W, vBias, hBias):
        print "it isn`t working"

def OpenRBM(string):
    array = string.split('\n')
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

class RBM:

    def save(self):
        return self.bm.saveTo(self.visible, self.hidden, [self.hBias, self.vBias], [self.W])

    def __init__(self, hidden, visible, rnd, theanoRnd, W=None, hBias=None, vBias=None):
        bm = BM(theanoRnd)
        self.bm = bm
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
            hBiasInit = numpy.zeros(hidden, dtype=theano.config.floatX)

            hBias = theano.shared(hBiasInit, borrow = True)

        if vBias is None:
            vBiasInit = numpy.zeros(visible, dtype=theano.config.floatX)
            vBias = theano.shared(vBiasInit, borrow = True)

        self.W = W
        self.vBias = vBias
        self.hBias = hBias

    def gibbs(self, sample, countstep, function_mode):
        return self.bm.gibbs(sample, self.W, self.vBias, self.hBias, countstep, function_mode)

    def gibbs_function(self, sample, countstep, function_mode):
        res, updates = self.gibbs(sample, countstep, function_mode)
        Varibles = [sample]
        if isinstance(countstep, TensorVariable):
            Varibles.append(countstep)
        return theano.function(Varibles, res, updates=updates)

    def gibbs_function_from_rnd(self, countstep, function_mode):
        sample = self.bm.generateRandomsFromBinoZeroOne(T.ones_like(self.vBias) * 0.5)
        res, updates = self.gibbs(sample, countstep, function_mode)
        Varibles = []
        if isinstance(countstep, TensorVariable):
            Varibles.append(countstep)
        return theano.function(Varibles, res, updates=updates)

    def free_energy(self, samples, countstep, function_mode):
        dream, update = self.gibbs(samples, countstep, function_mode)
        energy = T.mean(self.bm.freeEnergy(samples, self.W, self.vBias, self.hBias)) - \
                 T.mean(self.bm.freeEnergy(dream, self.W, self.vBias, self.hBias))
        return energy, update

    def gradient(self, samples, countstep, function_mode):
        dream, update = self.gibbs(samples, countstep, function_mode)
        energy = T.mean(self.bm.freeEnergy(samples, self.W, self.vBias, self.hBias)) - \
                 T.mean(self.bm.freeEnergy(dream, self.W, self.vBias, self.hBias))
        gradBlock = [self.W, self.hBias, self.vBias]
        grad = theano.grad(energy, gradBlock, [samples, dream])
        return energy, grad, gradBlock, update

    def grad_function(self, samples, countStep, function_mode, learning_rate, regularization = 0):
        energy, grad, gradBlock, update = self.gradient(samples, countStep, function_mode)
        for u, v in zip(gradBlock, grad):
            update[u] = u - learning_rate * (v + u * regularization)
                                             #+ 0.421 * u)
                                             # 0.01239 * u)
                                             # 0.321 * u)
                                             # 0.123 * u)
                                             # 0.05721 * u) # + 0.0923 * u)
        Varibles = [samples]
        if isinstance(countStep, TensorVariable):
            Varibles.append(countStep)
        if isinstance(learning_rate, TensorVariable):
            Varibles.append(learning_rate)
        return theano.function(Varibles, energy, updates=update)


def OpenRTRBM(string):
    array = string.split('\n')
    parse_vector = lambda str: map(float, str.split(','))
    parse_matrix = lambda str: [parse_vector(substr) for substr in str.split('],')]
    hBias = theano.shared(numpy.asarray(parse_vector(array[2])), borrow=True)
    vBias = theano.shared(numpy.asarray(parse_vector(array[3])), borrow=True)
    h_lid_0 = theano.shared(numpy.asarray(parse_vector(array[4])), borrow=True)
    W = theano.shared(numpy.asarray(parse_matrix(array[5])), borrow=True)
    W1 = theano.shared(numpy.asarray(parse_matrix(array[6])), borrow=True)
    W2 = theano.shared(numpy.asarray(parse_matrix(array[7])), borrow=True)

    numpyRng = numpy.random.RandomState(1234)
    theanoRng = RandomStreams(numpyRng.randint(2 ** 30))
    return RTRBM(int(array[1]), int(array[0]), numpyRng, theanoRng, W, hBias, vBias, W1, W2, h_lid_0)

def createSimpleRTRBM(hidden, visible):
    numpyRng = numpy.random.RandomState(1234)
    theanoRng = RandomStreams(numpyRng.randint(2 ** 30))
    return RTRBM(hidden, visible, numpyRng, theanoRng)

class RTRBM:
    def __init__(self, hidden, visible, rnd, theanoRnd, W=None, hBias=None, vBias=None, W1=None, W2=None, h_lid_0=None):
        bm = BM(theanoRnd)
        self.bm = bm
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
            vBiasInit = numpy.zeros(visible, dtype=theano.config.floatX)
            vBias = theano.shared(vBiasInit, borrow = True)

        if h_lid_0 is None:
            h_lid_0_Init = numpy.zeros(hidden, dtype=theano.config.floatX)
            h_lid_0 = theano.shared(h_lid_0_Init, borrow = True)

        self.W = W
        self.W1 = W1
        self.W2 = W2
        self.vBiasbase = vBias
        self.hBiasbase = hBias
        self.h_lid_0 = h_lid_0

    def save(self):
        return self.bm.saveTo(self.visible, self.hidden, [self.hBiasbase, self.vBiasbase, self.h_lid_0], [self.W, self.W1, self.W2])

    # if sample is matrix, gibbs thinks that it is matrix [Time, Visible]
    def gibbs(self, sample, countStep, function_mode):
        # templates of Varibles for calculate h_lid by previous value
        calc_h_lid = lambda h_lid_old, sample: T.nnet.sigmoid(T.dot(sample, self.W) + self.hBiasbase)
        calc_hBiases = lambda h_lid: self.hBiasbase + T.dot(h_lid, self.W2.T)
        calc_vBiases = lambda h_lid: self.vBiasbase + T.dot(h_lid, self.W1.T)
        #   Parameter: countGibbsStep
        def gibbsSamplingForAllTime(sample, start_h_lid):
            def gibbsSamplingForOneStepTime(sample, h_lid):
                vBias = calc_vBiases(h_lid)
                hBias = calc_hBiases(h_lid)
                res, updates = self.bm.gibbs(sample, self.W, vBias, hBias, countStep, function_mode)
                return [res, calc_h_lid(start_h_lid, sample), vBias, hBias], updates
            [sample_res, hLids, vBiases, hBiases], updates = theano.scan(gibbsSamplingForOneStepTime, sequences=sample, outputs_info=[None, start_h_lid, None, None])
            return sample_res, hLids, vBiases, hBiases, updates
        # usual gibbs-sampling
        if len(sample.broadcastable) == 2:
        #     matrix! it is one object
            res, hLids, vBiases, hBiases, updates = gibbsSamplingForAllTime([sample], self.h_lid_0)
            hLids = T.concatenate([[self.h_lid_0], hLids[0:-1]])
            return res, hLids, updates, vBiases, hBiases
        else:
            new_dim = T.cast(sample.shape[0], 'int32');
            my_sample = T.transpose(sample, (1, 0, 2))
            h_lids_start = T.reshape(T.repeat(self.h_lid_0, new_dim), (self.hidden, new_dim)).T
            res, hLids, vBiases, hBiases, updates = gibbsSamplingForAllTime(my_sample, h_lids_start)
            res = T.transpose(res, (1, 0, 2))
            hLids = T.concatenate([[h_lids_start], hLids[0:-1]])
            hLids = T.transpose(hLids, (1, 0, 2))
            vBiases = T.transpose(vBiases, (1, 0, 2))
            hBiases = T.transpose(hBiases, (1, 0, 2))
            return res, hLids, updates, vBiases, hBiases

    def gradient(self, sample, countStep, function_mode):
        # GradientBlock = [energy, gradUByW1, gradUByW2, gradUByhBias, gradUByvBias, gradUByW, gradH_lid0]
        def GradientForOneObject(sample, dream, h_lids, vBias, hBias):
            energy = self.bm.freeEnergy(sample, self.W, vBias, hBias) - self.bm.freeEnergy(dream, self.W, vBias, hBias)
            # energy = T.sum(energy)
            grad = theano.grad(energy, [self.W, vBias, hBias], consider_constant=[sample, dream])
            gradUByW1 = T.outer(grad[1], h_lids);
            gradUByW2 = T.outer(grad[2], h_lids);
            gradUByhBias = (grad[2]);
            gradUByvBias = (grad[1]);
            gradUByW = grad[0];
            gradHLid0 = (h_lids);
            return [energy, gradUByW1, gradUByW2, gradUByhBias, gradUByvBias, gradUByW, gradHLid0]
            # return [energy, grad[0], grad[1], grad[2], gradUByW1, gradUByW2, gradUByhBias]

        def GradientForOneTimeAutoGenerate(sample):
            # input one object = [time * visible]
            dream, h_lids, update, vBiases, hBiases = self.gibbs(sample, countStep, function_mode)
            res, update1 = theano.scan(GradientForOneObject, sequences=[sample, dream, h_lids, vBiases, hBiases])
            res2 = [T.sum(grad, axis=0) for grad in res]
            return res2, update + update1

        def GradientForAllTime(samples):
            Q, updates = theano.scan(GradientForOneTimeAutoGenerate, sequences=samples)
            meanQ = [T.mean(grad, axis=0) for grad in Q]
            return meanQ, updates

        GradientBlock = [self.W1, self.W2, self.hBiasbase, self.vBiasbase, self.W, self.h_lid_0]
        output, updates = GradientForAllTime(sample);
        output[-1] = self.h_lid_0 - output[-1]
        return output[0], GradientBlock, output[1:], updates

    def grad_function(self, countStep, learningRate, function_mode):
        samples = T.tensor3()
        energy, gb, grad, upd = self.gradient(samples, countStep, function_mode)
        self.bm.addGradientToUpdate(upd, gb, grad, learningRate)
        Varibles = [samples]
        if isinstance(countStep, TensorVariable):
            Varibles.append(countStep)
        if isinstance(learningRate, TensorVariable):
            Varibles.append(learningRate)
        return theano.function(Varibles, energy, updates=upd)
