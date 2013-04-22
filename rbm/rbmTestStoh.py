__author__ = 'gavr'

from rbmTest import *
from StringIO import StringIO

def function_make_dir(imsize, bkg, ci, cg, lr, lm, h, sW, r, tb, nreg):
    string = StringIO()
    string.write('9Stoh_')
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


app, data = rbmGenerateClocks()
hidden = [50]
tbs = [len(data), 2, 5, 15, 30]
regularization = [0]
gibbs = [10]
ms = [MODE_WITH_COIN_EXCEPT_LAST, MODE_WITHOUT_COIN]
all_new_reg = [1, 2, 10, 0]
for tb in tbs:
    for h in hidden:
        for m in ms:
            for x in regularization:
                for g in gibbs:
                    for new_reg in all_new_reg:
                        function_make_dir(30, False, 2001, g, 0.01, m, h, 1, x, tb, new_reg)
                        rbmStohasticGradientTest(countGibbs = g, outputEveryIteration = 400, countIteration=2001, data=data, appearance=app, trainBlock=tb, hidden=h, learningMode=m, regularization=x, newReg=new_reg)