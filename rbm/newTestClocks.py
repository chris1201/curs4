from newRTRBM import *
from utils import *
from rbmTest import *
import matplotlib.pyplot as plt

def test(hidden, l1, l2, widthsLines = [1]):
    rbm = createSimpleRBM(hidden * hidden, 900)
    # constuct data for multiple clocks
    data = []
    for width in widthsLines:
        app, data1 = rbmGenerateClocks(30, width)
        for x in data1: 
            data.append(x)
    func2 = rbm.grad2_function(T.matrix(), 0.001, 0, 0)
    func1 = rbm.grad_function(T.matrix(), 10, MODE_WITHOUT_COIN, 0.01, l2, 0, l1)

    countGibbs = 10
    numOutputRandom = 10
    m = T.matrix()
    sample = rbm.bm.generateRandomsFromBinoZeroOne(T.reshape(T.repeat(T.ones_like(rbm.vBias) * 0.5, numOutputRandom), (numOutputRandom, len(data[0]))))

    res, updates = rbm.bm.gibbs_all(sample, rbm.W, rbm.vBias, rbm.hBias, 10 + 11, MODE_WITHOUT_COIN) 

    rnd_gibbs = theano.function([], T.concatenate([[sample], res]), updates=updates)

    res, updates = rbm.bm.gibbs_all(m, rbm.W, rbm.vBias, rbm.hBias, countGibbs + 1, MODE_WITHOUT_COIN)
    data_gibbs = theano.function([m], T.concatenate([[m], res]), updates=updates)
    saveOutput = lambda x: makeAnimImageFromMatrixImages( convertProbabilityTensorToImages(app, x))
    eu = []
    loglh = []
    for i in range(30):
        for i1 in range(400):
            r = func2(data)
            eu.append(r)
        print i, ': ', r 
        for i1 in range(400):
            r = func1(data)
            loglh.append(r)
        print i, ': ', r
    
    wt = createFromWeightsImage(theano.function([], rbm.W.T)(), hidden, hidden, (30, 30))
    w = createFromWeightsImage(theano.function([], rbm.W)(), 30, 30, (hidden, hidden))
    d = saveOutput(data_gibbs(data))
    r = saveOutput(rnd_gibbs())
    setCurrentDirectory('h' + str(hidden) + '_l1' + str(l1) + '_l2' + str(l2))
    saveImage(wt, 'wt')
    saveImage(w, 'w')
    saveImage(d, 'data')
    saveImage(r, 'random')
    plt.clf()
    plt.cla()
    plt.subplot(121)
    for i in range(100):
        eu[i] = 0
    plt.plot(eu)
    plt.subplot(122)
    plt.plot(loglh, 'r')
    plt.savefig(ccd.currentDirectory + 'progress.png')
    saveData(rbm.save())
    saveImage(createFromWeightsImage(theano.function([], T.dot(rbm.W, rbm.W.T))(), 30, 30, (30, 30)), 'wwt')

def testRTRBM(hidden, l1, l2, widthLines = [1]):
    
