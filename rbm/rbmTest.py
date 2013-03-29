__author__ = 'gavr'

from newRTRBM import *
from clocks import *
from utils import *
from tictoc import tic, toc
from StringIO import StringIO

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
            regularization = 0, prefixName = ''):
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
    SetGreyAsBlack()
    if NotDrawBackGround:
        SetDontDrawBlackContour()
    else:
        SetDrawBlackContour()
    SetSecWidth(secWidth)
    dials = DrawDials(Tick(0, 0, 0), Tick(59, 0, 0),  imageSize)
    appearance = dials[0]
    dataPrime = [convertImageToVector(element) for element in dials]
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
    print "learning time: ", toc()
    saveData(rbm.save())

# rbmTest(countIteration=3201, learningMode=MODE_WITH_COIN_EXCEPT_LAST, countGibbs=10, hiddenVaribles=400, prefixName='1')

countIterations = [3201]
modes = [MODE_WITHOUT_COIN, MODE_WITH_COIN, MODE_WITH_COIN_EXCEPT_LAST, MODE_WITHOUT_COIN_EXCEPT_LAST]
countGibbs = [2, 5, 10, 20, 40, 80]
hidden = [50, 100, 200, 400, 600, 900]
secWidth = [1, 2, 3]
regularization = [0, 0.001, 0.01, 0.005]
NotDrawBackGround = [False, True]

for ci in countIterations:
    for cg in countGibbs:
        for m in modes:
            for h in hidden:
                for sw in secWidth:
                    for r in regularization:
                        for ndbg in NotDrawBackGround:
                            rbmTest(\
                                countIteration=ci, \
                                learningMode=m, \
                                countGibbs=cg, \
                                hiddenVaribles=h, \
                                NotDrawBackGround=ndbg, \
                                secWidth=sw, \
                                regularization=r,\
                                prefixName='2')

