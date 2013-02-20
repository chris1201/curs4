from re import template

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
    def __init__(self, hidden, visible, rnd, theanoRnd, W = None, hBias = None, vBias = None, W1 = None, W2 = None):
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

        # save in class
        self.W = W
        self.W1 = W1
        self.W2 = W2
        self.vBias = vBias
        self.hBias = hBias
        # varibles
        data = T.tensor3()
        countGibbsSteps = T.iscalar()
        lengthOfSequence = T.iscalar()
        learningRate = T.fscalar()
        # create h_lid_start
        def create_h_lid_start(samples, isOneTime = None):
            if samples.ndim == 3:
                a, b, c = samples.shape
                return T.zeros((c, hidden))
            else:
                if isOneTime is None:
                    return T.zeros_like(hBias)
                else:
                    if isOneTime:
                        a, b = samples.shape
                        return T.zeros(b, hidden)
                    else:
                        return T.zeros_like(hBias)
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
        # function for generate initial state for visible varibles
        #   Parameter: lengthOfSequence
        generateRandomVisiblesForOneTime_format = lambda: theanoRnd.binomial(size=vBias.shape, n=1, p=T.ones_like(vBias) * 0.5, dtype='floatX')
        generateRandomVisibles_format, unusedUpdates = theano.scan(generateRandomVisiblesForOneTime_format, n_steps=lengthOfSequence)
        def generateRandomVisibles_format2(length):
            rnd, unusedUpdates = theano.scan(generateRandomVisiblesForOneTime_format, n_steps=length)
            return rnd

        # templates of Varibles for calculate h_lid by previous value
        calc_h_lid = lambda h_lid_old, sample: T.nnet.sigmoid(T.dot(sample, W) + T.dot(h_lid_old, W1) + hBias)
        calc_hBiases = lambda h_lid: hBias + T.dot(W2, h_lid)
        calc_vBiases = lambda h_lid: vBias + T.dot(W1, h_lid)
        # Make For All input Images(sample) gibbs-sampling for one-time
        #   Parameter: countGibbsStep
        def gibbsSamplingForOneStepTime(sample, h_lid):
            res, updates = theano.scan(gibbsOne_format, outputs_info=sample, non_sequences=[calc_vBiases(h_lid), calc_hBiases(h_lid)], n_steps=countGibbsSteps)
            res = res[-1]
            return [[res, calc_h_lid(h_lid, res)], updates]
        # Make gibbs-sampling for all-time
        #   Parameter: countGibbsStep
        def gibbsSamplingForAllTime(sample, start_h_lid):
            [samp_res, hlids], updates = theano.scan(gibbsSamplingForOneStepTime, sequences=sample, outputs_info=[None, start_h_lid])
            return samp_res, hlids, updates
        def calc_Energy(sample, vBiases, hBiases):
            q = hBiases + T.dot(sample, W)
            # energy for One Time and for all objects in sample
            energyOne = T.dot(vBiases, sample) + T.sum(T.log(1 + T.exp(q)))
            energy = T.sum(energyOne)
            return energy, vBiases, hBiases
        # Calc h_lids
        def calc_h_lids(sample, h_lid_start):
            h_lids, updates = theano.scan(calc_h_lid, sequences=sample, outputs_info=h_lid_start)
            return h_lids[0:-2], updates

        def calc_grad_Energy_For_One_Object(sample, h_lids, evaluatesample, updates):
            vBiases = calc_vBiases(h_lids)
            hBiases = calc_hBiases(h_lids)
            Pdata, _, _ = calc_Energy(sample, vBiases, hBiases)
            Pmodel, _, _ = calc_Energy(evaluatesample, vBiases, hBiases)
            P = Pdata - Pmodel
            grad = theano.grad(P, [W, vBiases, hBiases], consider_constant=[sample, evaluatesample, W1, W2, hBias, vBias])
            gradUByW1 = T.dot(h_lids, grad[1]);
            gradUByW2 = T.dot(grad[2], h_lids);
            gradUByhBias = T.sum(grad[1], axis=1);
            gradUByvBias = T.sum(grad[2], axis=1);
            gradUByW = grad[0];
            #block = [W1, W2, hBias, vBias, W]
            grad_block_return = [P, gradUByW1, gradUByW2, gradUByhBias, gradUByvBias, gradUByW]
            return grad_block_return, updates

        def calc_grad_Energy_For_One_Object_By_Data(sample):
            new_sample, _, updates = gibbsSamplingForAllTime(sample, create_h_lid_start(sample))
            h_lids, updates_h_lid = calc_h_lids(sample, create_h_lid_start(sample))
            return calc_grad_Energy_For_One_Object(sample, h_lids, new_sample, updates+updates_h_lid)

        def calc_grad_Energy_For_Input_Objects_By_Data(samples):
            #block = [W1, W2, hBias, vBias, W]
            #grad_block_return = [gradUByW1, gradUByW2, gradUByhBias, gradUByvBias, W]
            Q, updates = theano.scan(calc_grad_Energy_For_One_Object_By_Data, sequences=samples)
            meanQ = [T.mean(grad,axis=grad.ndim-1) for grad in Q]
            return meanQ, updates

        def calc_grad_Energy_For_One_Object_By_Rnd(sample):
            new_sample, _, updates = gibbsSamplingForAllTime(generateRandomVisibles_format2(sample.shape[1]), create_h_lid_start(sample))
            h_lids, updates_h_lid = calc_h_lids(sample, create_h_lid_start(sample))
            return calc_grad_Energy_For_One_Object(sample, h_lids, new_sample, updates+updates_h_lid)

        def calc_grad_Energy_For_Input_Objects_By_Rnd(samples):
            #block = [W1, W2, hBias, vBias, W]
            #grad_block_return = [gradUByW1, gradUByW2, gradUByhBias, gradUByvBias, W]
            Q, updates = theano.scan(calc_grad_Energy_For_One_Object_By_Rnd, sequences=samples)
            meanQ = [T.mean(grad,axis=grad.ndim-1) for grad in Q]
            return meanQ, updates

        shuffleData = data.dimshuffle(2, 1, 0)
        alls, upds = calc_grad_Energy_For_Input_Objects_By_Rnd(shuffleData)
        block = [W1, W2, hBias, vBias, W]
        for u, v in zip(block, alls[1:]):
            upds[u] = u - learningRate * v
        self.grad_step_rnd = theano.function([data, countGibbsSteps, learningRate], alls[0], updates=upds)
        print self.grad_step_rnd
        
        alls, upds = calc_grad_Energy_For_Input_Objects_By_Data(shuffleData)
        block = [W1, W2, hBias, vBias, W]
        for u, v in zip(block, alls[1:]):
            upds[u] = u - learningRate * v
        self.grad_step = theano.function([data, countGibbsSteps, learningRate], alls[0], updates=upds)
        print self.grad_step
# TODO save RTRBM
# TODO test RTRBM
# TODO Apply RTRBM for clocks.

def createSimpleRBM(hidden, visible):
    numpyRng = numpy.random.RandomState(1234)
    theanoRng = RandomStreams(numpyRng.randint(2 ** 30))
    return RTRBM(hidden, visible, numpyRng, theanoRng)

q = createSimpleRBM(10, 12)
