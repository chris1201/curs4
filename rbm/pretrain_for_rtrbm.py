__author__ = 'gavr'

from rbmTest import *
from StringIO import StringIO
from tictoc import tic, toc

def function_make_dir(imsize, bkg, ci, cg, lr, lm, h, sW, r, tb, nreg):
    string = StringIO()
    string.write('pre_train_for_rtrbm')
    string.write('IS_'+str(imsize))
    string.write('_bg_'+str(bkg))
    string.write('_ci_'+str(ci))
    string.write('_cg_'+str(cg))
    string.write('_lr_'+str(lr))
    string.write('_lm_'+MODE_NAMES[lm])
    string.write('_h_'+str(h))
    string.write('_sW_'+str(sW))
    string.write('_r_'+str(r))
    string.write('_tb_'+str(tb))
    string.write('_newReg'+str(nreg))

    setCurrentDirectory(string.getvalue())


ci = 50001
cg = 10
lr = 0.01
tb = 59
h1 = 11
h = h1 * h1
new_reg = 0.02
reg = 0
lm = MODE_WITHOUT_COIN
# function_make_dir(30, True, ci, cg, lr, lm, h, 1, reg, tb, new_reg)
setCurrentDirectory('86') # 67 --l1 = 0.05, 68 ---0.1, 69 --- 0.005, 70 - 0.001, 71 - 0.0005, 72 --- 50000Iter,
#  73-hidden 10, 74 --- hidden 36, 75 --- 0.005, 76 --- 10, 83 - stoh. 84 = full gradient
# 85 - 256. 86 - 121
trainRBM = 2
if trainRBM == 1:
    app, data = rbmGenerateClocks(30)
    rbm=rbmStohasticGradientTest(countGibbs = cg,
                             outputEveryIteration = 5000,
                             countIteration=ci,
                             data=data,
                             appearance=app,
                             trainBlock=tb,
                             hidden=h,
                             learningMode=lm,
                             regularization=reg,
                             newReg=new_reg,
                             learningRate=lr,
                             regL1=0.001)
    saveImage(createFromWeightsImage(theano.function([], rbm.W.T)(), h1, h1, (30, 30)), 'W')
    saveImage(createFromWeightsImage(theano.function([], rbm.W)(), 30, 30, (h1, h1)), 'Wh')


    saveImage(createFromWeightsImage(theano.function([], rbm.W.T + rbm.vBias)(), h1, h1, (30, 30)), 'W_b')
    saveImage(createFromWeightsImage(theano.function([], rbm.W + rbm.hBias)(), 30, 30, (h1, h1)), 'Wh_b')

    # saveImage(createFromWeightsImage(theano.function([], rbm.W.T + rbm.vBias)(), 11, 11, (30, 30)), 'W')
else:
    # setCurrentDirectory('64')
    rbm = OpenRBM(getStringData())
    saveImage(createFromWeightsImage(theano.function([], rbm.W.T)(), h1, h1, (30, 30)), 'W')
    saveImage(createFromWeightsImage(theano.function([], rbm.W)(), 30, 30, (h1, h1)), 'Wh')


    saveImage(createFromWeightsImage(theano.function([], rbm.W.T + rbm.vBias)(), h1, h1, (30, 30)), 'W_b')
    saveImage(createFromWeightsImage(theano.function([], rbm.W + rbm.hBias)(), 30, 30, (h1, h1)), 'Wh_b')

    setCurrentDirectory('8631GWReg1Iter30kElem2Stoh')
    # rtrbm = createSimpleRTRBM(h, 900)
    # rtrbm.vBiasbase = rbm.vBias
    # rtrbm.hBiasbase = rbm.hBias
    # rtrbm.W = rbm.W
    rtrbm = OpenRTRBM(getStringData('rtrbm5850000.txt'))
    setCurrentDirectory('8631GWReg1Iter30kElem2StohContinue1')

    # saveImage(createFromWeightsImage(theano.function([], rtrbm.W.T)(), 11, 11, (30, 30)), 'Wbegin')
    lm = MODE_WITHOUT_COIN
    func = rtrbm.grad_function(1, 0.01, lm, 1)

    # rtrbm_ci = 5001
    rtrbm_ci = 20001
    elementLength = 2

    app, dataPrime = rbmGenerateClocks(30)

    data = [dataPrime[idx:((idx + elementLength))] + (
        [] if (idx + elementLength) / len(dataPrime) == 0 else dataPrime[:((idx + elementLength) % len(dataPrime))])
            for idx in range(len(dataPrime))]
    print numpy.shape(data)

    funcSample = rtrbm.predict_function(True, 5, 1, lm)
    funcSample1 = rtrbm.predict_function(True, 5, 5, lm)
    funcSample2 = rtrbm.predict_function(True, 5, 10, lm)
    # funcSample3 = rtrbm.predict_function(True, 5, 15, lm)

    saveOutput = lambda x, name: \
        saveImage( \
            makeAnimImageFromMatrixImages( \
                convertProbabilityTensorToImages(app, x)),
            name)
    saveOutput(funcSample(data), 'rtrbm0')
    saveOutput(data, 'rtrbm_data')
    tic()
    tic()
    for iter in range(rtrbm_ci):
        # for inner_iter in range((len(data))):
        #     x = func([data[inner_iter]])
        x = func(data)
        if (iter % 2500 == 0):
            print 'output, x:', x, 'time', toc()
            tic()
            saveOutput(funcSample(data), 'rtrbm_output' + str(iter))
            saveOutput(funcSample1(data), 'rtrbm1_output' + str(iter))
            saveOutput(funcSample2(data), 'rtrbm2_output' + str(iter))
            # saveOutput(funcSample3(data), 'rtrbm3_output' + str(iter))
            saveData(rtrbm.save(), 'rtrbm' + str(idx) + str(iter) + '.txt')
    toc()
    print 'time', toc()
    # saveImage(createFromWeightsImage(theano.function([], rtrbm.W.T)(), 11, 11, (30, 30)), 'Wend')
