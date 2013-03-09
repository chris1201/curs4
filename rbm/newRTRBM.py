__author__ = 'gavr'

from theano.gof.python25 import OrderedDict
import theano.updates
import numpy
from math import sqrt
import theano
import theano.tensor as T
from numpy.oldnumeric.random_array import random_integers
from theano.tensor.shared_randomstreams import RandomStreams
from tictoc import tic, toc
MODE_WITHOUT_COIN = 0;
MODE_WITH_COIN = 1;

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
                self.generateRandomsFromBinoZeroOne(self.computeProbabilityHbyV(sample, W, hBias)), W, vBias))
    def gibbs(self, sample, W, vBias, hBias, countSteps, function_mode):
        gibbsOne_format = lambda sample: self.list_function_for_gibbs[function_mode](sample, W, vBias, hBias);
        format, updates = theano.scan(fn=gibbsOne_format, \
                                      outputs_info=sample, \
                                      n_steps=countSteps)
        return format[-1], updates
    def freeEnergy(self, sample, W, vBias, hBias):
        xdotw_plus_bias = T.dot(sample, W) + hBias
        xdotvbias = T.dot(sample, vBias)
        sum_log = T.sum(T.log(1 + T.exp(xdotw_plus_bias)), axis=0)
        return -sum_log - xdotvbias
    def addGradientToUpdate(self, update, gradVaribles, grad, learningRate):
        for u, v in zip(gradVaribles, grad):
            update[u] = u - learningRate * v

class RTRBM:
    def __init__(self, hidden, visible, rnd, theanoRnd, W = None, hBias = None, vBias = None, W1 = None, W2 = None, h_lid_0 = None):
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
            vBiasInit = numpy.zeros(visible)
            vBias = theano.shared(vBiasInit, borrow = True)

        if h_lid_0 is None:
            h_lid_0_Init = numpy.zeros(hidden)
            h_lid_0 = theano.shared(h_lid_0_Init, borrow = True)

        self.W = W
        self.W1 = W1
        self.W2 = W2
        self.vBias = vBias
        self.hBias = hBias
        self.h_lid_0 = h_lid_0

    def calcVHBiases(self, h_lid):
        hBias = self.hBias + T.dot(h_lid, self.W2.T)
        vBias = self.vBias + T.dot(h_lid, self.W1.T)
        return vBias, hBias

    def calcHLid(self, h_lid_prev, sampleV):
        return T.nnet.sigmoid(T.dot(sampleV, self.W) + T.dot(self.W2, h_lid_prev) + self.hBias)

    def gibbs(self, sample, countStep, function_mode):
        def alone_gibbs(sample, h_lid):
            v, h = self.calcVHBiases(h_lid)
            result, updates = self.bm.gibbs(sample, self.W, v, h, countStep, function_mode)
            return [result, self.calcHLid(h_lid, sample), updates]
#        [result, h_lids], updates = theano.scan(alone_gibbs, outputs_info=[None, self.h_lid_0],

visible = 900
hidden = 100
numpyRng = numpy.random.RandomState(1234)
theanoRng = RandomStreams(numpyRng.randint(2 ** 30))
bm = BM(theanoRng)
WInit = numpy.asarray(numpyRng.uniform(
  low=-4 * sqrt(6. / (hidden + visible)),
  high=4 * sqrt(6. / (hidden + visible)),
  size=(visible, hidden)),
  dtype=theano.config.floatX)
W = theano.shared(WInit, borrow = False)
hBiasInit = numpy.zeros(hidden, dtype=theano.config.floatX)
hBiasbase = theano.shared(hBiasInit, borrow = True)
vBiasInit = numpy.zeros(visible)
vBiasbase = theano.shared(vBiasInit, borrow = True)
W1Init = numpy.asarray(numpyRng.uniform(
  low=-4 * sqrt(6. / (hidden + visible)),
  high=4 * sqrt(6. / (hidden + visible)),
  size=(visible, hidden)),
  dtype=theano.config.floatX)
W1 = theano.shared(W1Init, borrow = False)
W2Init = numpy.asarray(numpyRng.uniform(
  low=-4 * sqrt(6. / (hidden + hidden)),
  high=4 * sqrt(6. / (hidden + hidden)),
  size=(hidden, hidden)),
  dtype=theano.config.floatX)
W2 = theano.shared(W2Init, borrow = False)
h_lid_0_Init = numpy.zeros(hidden)
h_lid_0 = theano.shared(h_lid_0_Init, borrow = True)
def calcVHBiases(h_lid):
    hBias = hBiasbase + T.dot(h_lid, W2.T)
    vBias = vBiasbase + T.dot(h_lid, W1.T)
    return vBias, hBias

def calcHLid(h_lid_prev, sampleV):
    return T.nnet.sigmoid(T.dot(sampleV, W) + T.dot(h_lid_prev, W2.T) + hBiasbase)

def gibbs(sample, countStep, function_mode):
    def alone_gibbs(sample, h_lid):
        v, h = calcVHBiases(h_lid)
        result, updates = bm.gibbs(sample, W, v, h, countStep, function_mode)
        return [[result, v, h, calcHLid(h_lid, sample)], updates]

    new_dim = T.cast(sample.shape[0], 'int32');
    h_lids = T.reshape(T.repeat(h_lid_0, new_dim), (hidden, new_dim)).T
    data = T.transpose(sample, (1, 0, 2))
    [result, vBiases, hBiases, h_lids], updates = theano.scan(alone_gibbs, outputs_info=[None, None, None, h_lids], sequences=data)
    return result, h_lids, vBiases, hBiases, updates

def energy(sample, countStep, function_mode):
    dream, h_lids, vBiases, hBiases, updates = gibbs(sample, countStep, function_mode)
    dream = T.transpose(dream, (1, 0, 2))
    vBiases = T.transpose(vBiases, (1, 0, 2))
    hBiases = T.transpose(hBiases, (1, 0, 2))
    h_lids = T.transpose(h_lids, (1, 0, 2))
    def freeEnergy(sample, W, vBias, hBias):
        xdotw_plus_bias = T.tensordot(sample, W, [[2], [0]]) + hBias
        xdotvbias = T.tensordot(sample, vBias, [[1, 2], [1, 2]])
        xdotvbias = T.diagonal(xdotvbias)
        sum_log = T.sum(T.log(1 + T.exp(xdotw_plus_bias)), axis=(1, 2))
        energies = T.mean(-sum_log - xdotvbias)
        return energies
    def cleverSigma(W, sample, hBias):
        return T.nnet.sigmoid(T.tensordot(sample, W, [[2], [0]]) - hBias)
    logP = freeEnergy(sample, W, vBiases, hBiases) - freeEnergy(dream, W, vBiases, hBiases)
    gradByvBiases = sample - dream
    partSample = cleverSigma(W, sample, hBiases)
    partDream = cleverSigma(W, dream, hBiases)
    gradByhBiases = partDream - partSample
    # (3, 10, 5) * (3, 10, 15) = (3, 5, 15)
    gradByW = T.tensordot(partDream, dream, [[1], [1]]) - T.tensordot(partSample, sample, [[1], [1]])
    gradByW = T.mean(T.diagonal(gradByW, axis1=0, axis2=2), axis=2)
    gradByW1 = T.tensordot(gradByhBiases, h_lids, [[1], [1]])
    gradByW1 = T.mean(T.diagonal(gradByW1, axis1=0, axis2=2), axis=2)
    gradByW2 = T.tensordot(gradByvBiases, h_lids, [[1], [1]])
    gradByW2 = T.mean(T.diagonal(gradByW2, axis1=0, axis2=2), axis=2)
#    gradByHbias =
#    gradByW = T.mean(T.sum(T.tensordot(partDream, dream) - T.tensordot(partSample, sample), axis=1), axis=0)
#    gradblock = [W, hBiases, vBiases]
    grad = [gradByhBiases, gradByvBiases, gradByW, gradByW1, gradByW2]
#    grad = theano.grad(logP, gradblock, consider_constant=[dream, sample])
    return logP, grad, updates
# TODO question 1: h_lids in gibbs not true, remove last element and add first element
# 
m = T.tensor3()
#m2 = T.matrix()
#res, upds = bm.gibbs(m, W, vBiasbase, hBiasbase, 5, 0)
#
r1, r2, r3, r4, u = gibbs(m, 5, 0)
#new_dim = T.cast(m.shape[0], 'int32');
#h_lids = T.reshape(T.repeat(h_lid_0, new_dim), (hidden, new_dim)).T
#a, b = theano.function([m], calcVHBiases(h_lids))(numpy.zeros((3, 10, visible)))
#c = theano.function([m, m2], calcHLid(h_lids, m2))(numpy.zeros((3, 10, visible)), numpy.zeros((3, visible)))
#print numpy.shape(a), numpy.shape(b), numpy.shape(c)
#[result, h_lids], updates = theano.scan(alone_gibbs, outputs_info=[None, h_lids], sequences=sample)

#a, u = energy(m, 5, 0)
#print theano.gof.cmodule.python_int_bitwidth()
#print theano.configdefaults.cpuCount()
f = theano.function([m], [r1, r2, r3, r4], updates = u)
#rr1, rr2, rr3, rr4 = f(numpy.zeros((3, 10, visible)))
#print numpy.shape(rr1), numpy.shape(rr2), numpy.shape(rr3), numpy.shape(rr4)
#zeromat = numpy.zeros((3, 10, visible))
#res = T.tensordot(m, W, [[2], [0]])
#print numpy.shape(theano.function([m], res)(zeromat))

#m3 = T.transpose(r3, (1, 0, 2))
#res0, _ = theano.scan(fn=lambda x, y: T.tensordot(x, y.T), sequences=[m, m3])
#f3 = theano.function([m], res0)

#res = T.tensordot(m, r3, [[1, 2], [0, 2]])
#res1 = T.diagonal(res)
#print res1
#f1  = theano.function([m], res1);
#f2 = theano.function([m], res);
#tic();
#temp = f1(zeromat)
#print numpy.shape(temp), numpy.shape(temp[0]), numpy.shape(temp[1]), '   ', toc()
#tic();
#print numpy.shape(f2(zeromat)), '   ', toc()
#tic();
#print numpy.shape(f3(zeromat)), '   ', toc()

#a, g, u = energy(m, 5, 0)
#tic()
#f = theano.function([m], [a, g[0], g[1], g[2], g[3], g[4]], updates=u)
#print toc()
#tic()
#e, g1, g2, g3, g4, g5 = f(numpy.zeros((3, 10, visible)))
#print e, numpy.shape(g1), numpy.shape(g2), numpy.shape(g3), numpy.shape(g4), numpy.shape(g5),toc()
#print numpy.shape(a), numpy.shape(b)
#
#new_dim = T.cast(m.shape[0], 'int32');
#z = T.reshape(T.repeat(m[0][0], new_dim), (visible, new_dim)).T
#t3 = numpy.asarray(numpyRng.uniform(
#  low=-4 * sqrt(6. / (hidden + visible)),
#  high=4 * sqrt(6. / (hidden + visible)),
#  size=(3, 10, visible)),
#  dtype=theano.config.floatX)
#q = T.transpose(m, (1, 0, 2))
#print numpy.shape(theano.function([m], q[0])(t3))