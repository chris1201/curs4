from appindicator._appindicator import app_indicator_get_type

__author__ = 'gavr'

from newRTRBM import *
from clocks import *
from utils import *
from tictoc import tic, toc
from StringIO import StringIO
import random

def rbmGenerateClocks(imageSize = 30, NotDrawBackGround = False, secWidth = 1):
    SetGreyAsBlack()
    if NotDrawBackGround:
        SetDontDrawBlackContour()
    else:
        SetDrawBlackContour()
    SetSecWidth(secWidth)
    dials = DrawDials(Tick(0, 0, 0), Tick(59, 0, 0),  imageSize)
    appearance = dials[0]
    dataPrime = [convertImageToVector(element) for element in dials]
    return appearance, dataPrime

def rbmStohasticGradientTest(countIteration = 2401,
                             countGibbs = 5,
                             learningRate = 0.01,
                             learningMode = MODE_WITHOUT_COIN,
                             outputEveryIteration = 100,
                             trainBlock = 100,
                             data = None,
                             regularization = 0,
                             numOutputRandom = 20,
                             hidden = 50,
                             appearance = None, newReg = 0.01):
    rbm = createSimpleRBM(hidden, len(data[0]))
    m = T.matrix()
    n = T.iscalar()
    s = T.fscalar()
    v = T.vector()
    reg = T.scalar()
    print "start create learning function", tic()
    grad_func = rbm.grad_function(m, countGibbs, learningMode, learningRate,
                                  reg, newReg)
    print "learning function has been built: ", toc()
    print "start contruct gibbs function"
    tic()
    sample = rbm.bm.generateRandomsFromBinoZeroOne(
        T.reshape(
            T.repeat(T.ones_like(rbm.vBias) * 0.5, numOutputRandom),
            (numOutputRandom, len(data[0]))))
    res, updates = rbm.bm.gibbs_all(sample, rbm.W, rbm.vBias, rbm.hBias, countGibbs + 1, learningMode)

    rnd_gibbs = theano.function([], T.concatenate([[sample], res]), updates=updates)

    res, updates = rbm.bm.gibbs_all(m, rbm.W, rbm.vBias, rbm.hBias, countGibbs + 1, learningMode)
    data_gibbs = theano.function([m], T.concatenate([[m], res]), updates=updates)
    print "Constructed Gibbs function: ", toc()
    saveOutput = lambda x, name: \
        saveImage( \
            makeAnimImageFromMatrixImages( \
                convertProbabilityTensorToImages(appearance, x)),
            name)
    print "Start Learn"
    tic()
    tic()
    random.shuffle(data)
    for idx in range(countIteration):
        for iteration in range(len(data) / trainBlock + 1):
            dataPrime = data[iteration * trainBlock: (iteration + 1) * trainBlock]
            if len(dataPrime) > 0:
                res = grad_func(dataPrime, regularization + (len(data) - len(dataPrime)) * 0.00 / len(data))
        if idx % outputEveryIteration == 0:
            print res, ' time: ', toc()
            tic()
            saveOutput(rnd_gibbs(), 'random' + str(idx) + '_' + str(iteration))
            saveData(rbm.save(), str(idx) + '.txt')
            saveOutput(data_gibbs(data), 'data' + str(idx) + '_' + str(iteration))

    toc()
    print "learning time: ", toc()
    saveData(rbm.save())
    return rbm


def rbmTest(imageSize = 30, \
            NotDrawBackGround = False, \
            countIteration = 2401, \
            outputEveryIteration = 100, \
            countGibbs = 10,
            learningRate = 0.01,
            hiddenVaribles = 100,
            secWidth = 1,
            learningMode = MODE_WITHOUT_COIN,
            numOutputRandom = 10,
            regularization = 0,
            prefixName = '',
            dataFromOut = None):
    string = StringIO()
    string.write(prefixName)
    string.write('IS_'+str(imageSize))
    string.write('_bg_'+str(NotDrawBackGround))
    string.write('_ci_'+str(countIteration))
    string.write('_cg_'+str(countGibbs))
    string.write('_lr_'+str(learningRate))
    string.write('_lm_'+MODE_NAMES[learningMode])
    string.write('_h_'+str(hiddenVaribles))
    string.write('_sW_'+str(secWidth))
    string.write('_r_'+str(regularization))
    setCurrentDirectory(string.getvalue())
    if dataFromOut is None:
        SetGreyAsBlack()
        if NotDrawBackGround:
            SetDontDrawBlackContour()
        else:
            SetDrawBlackContour()
        SetSecWidth(secWidth)
        dials = DrawDials(Tick(0, 0, 0), Tick(59, 0, 0),  imageSize)
        appearance = dials[0]
        dataPrime = [convertImageToVector(element) for element in dials]
    else:
        dataPrime = dataFromOut
        appearance = Image.new('F', size=(imageSize, imageSize))
    # save(data)
    rbm = createSimpleRBM(hiddenVaribles, imageSize * imageSize)
    m = T.matrix()
    n = T.iscalar()
    s = T.fscalar()
    v = T.vector()
    print "start create learning function", tic()
    grad_func = rbm.grad_function(m, countGibbs, learningMode, learningRate, regularization)
    print "learning function has been built: ", toc()
    print "start contruct gibbs function"
    tic()
    sample = rbm.bm.generateRandomsFromBinoZeroOne(
        T.reshape(
            T.repeat(T.ones_like(rbm.vBias) * 0.5, numOutputRandom),
            (numOutputRandom, imageSize * imageSize)))
    res, updates = rbm.bm.gibbs_all(sample, rbm.W, rbm.vBias, rbm.hBias, countGibbs, MODE_WITHOUT_COIN)

    rnd_gibbs = theano.function([], T.concatenate([[sample], res]), updates=updates)

    res, updates = rbm.bm.gibbs_all(m, rbm.W, rbm.vBias, rbm.hBias, countGibbs, MODE_WITHOUT_COIN)
    data_gibbs = theano.function([m], res, updates=updates)
    print "Constructed Gibbs function: ", toc()
    saveOutput = lambda x, name: \
        saveImage(\
            makeAnimImageFromMatrixImages(\
                convertProbabilityTensorToImages(appearance, x)),
            name)
    print "Start Learn"
    tic()
    tic()
    for idx in range(countIteration):
        res = grad_func(dataPrime)
        if idx % outputEveryIteration == 0:
            saveOutput(data_gibbs(dataPrime), 'data' + str(idx))
            saveOutput(rnd_gibbs(), 'random' + str(idx))
            print idx, res, toc()
            tic()
    toc()
    print "learning time: ", toc()
    saveData(rbm.save())

# rbmTest(countIteration=3201, learningMode=MODE_WITH_COIN_EXCEPT_LAST, countGibbs=10, hiddenVaribles=400, prefixName='1')
if __name__ == '__main__':

    countIterations = [5000]
    # MODE_WITHOUT_COIN, MODE_WITH_COIN, MODE_WITH_COIN_EXCEPT_LAST,
    modes = [MODE_WITHOUT_COIN]#, MODE_WITH_COIN_EXCEPT_LAST]
    countGibbs = [5]#[10, 20, 40]#[2, 5, 10, 20, 40, 80]
    hidden = [50]#[50, 100, 200, 400]#, 600, 900]
    secWidth = [3]#%, 2, 3]
    regularization = [0.001]#, 0.005]
    NotDrawBackGround = [False]#, True]
    learningRates = [0.01]

    for ci in countIterations:
        for cg in countGibbs:
            for m in modes:
                for h in hidden:
                    for sw in secWidth:
                        for r in regularization:
                            for ndbg in NotDrawBackGround:
                                for lr in learningRates:
                                    rbmTest(\
                                        countIteration=ci, \
                                        learningMode=m, \
                                        countGibbs=cg, \
                                        hiddenVaribles=h, \
                                        NotDrawBackGround=ndbg, \
                                        secWidth=sw, \
                                        regularization=r,\
                                        learningRate=lr,\
                                        prefixName='2')

