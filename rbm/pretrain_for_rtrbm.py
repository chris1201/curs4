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


ci = 20001
cg = 10
lr = 0.01
tb = 59
h = 100
new_reg = 0.1
reg = 0
lm = MODE_WITH_COIN_EXCEPT_LAST
# function_make_dir(30, True, ci, cg, lr, lm, h, 1, reg, tb, new_reg)
setCurrentDirectory('59')
trainRBM = 1
if trainRBM == 1:
    app, data = rbmGenerateClocks()
    rbm=rbmStohasticGradientTest(countGibbs = cg,
                             outputEveryIteration = 2000,
                             countIteration=ci,
                             data=data,
                             appearance=app,
                             trainBlock=tb,
                             hidden=h,
                             learningMode=lm,
                             regularization=reg,
                             newReg=new_reg,
                             learningRate=lr)
    saveImage(createFromWeightsImage(theano.function([], rbm.W.T)(), 8, 8, (30, 30)), 'W')
else:
    #rbm = OpenRBM(getStringData())
    rtrbm = createSimpleRTRBM(h, 900)
    #rtrbm.vBiasbase = rbm.vBias
    #rtrbm.hBiasbase = rbm.hBias
    #rtrbm.W = rbm.W

    func = rtrbm.grad_function(5, 0.01, lm, 100)

    rtrbm_ci = 40001
    elementLength = 2

    app, dataPrime = rbmGenerateClocks()

    data = [dataPrime[idx:((idx + elementLength))] + (
        [] if (idx + elementLength) / len(dataPrime) == 0 else dataPrime[:((idx + elementLength) % len(dataPrime))])
            for idx in range(len(dataPrime))]
    print numpy.shape(data)

    funcSample = rtrbm.predict_function(True, 5, 1, lm)
    funcSample1 = rtrbm.predict_function(True, 5, 2, lm)

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
#        for inner_iter in range(int(len(data) / 6)):
#            x = func([data[inner_iter]])
        x = func(data)
        if (iter % 1000 == 0):
            print 'output, x:', x, 'time', toc()
            tic()
            saveOutput(funcSample(data), 'rtrbm_output' + str(iter))
            saveOutput(funcSample1(data), 'rtrbm1_output' + str(iter))
            saveData(rtrbm.save(), 'rtrbm' + str(idx) + '.txt')
    toc()
    print 'time', toc()
    saveImage(createFromWeightsImage(theano.function([], rtrbm.W.T)(), 10, 10, (30, 30)), 'W')
