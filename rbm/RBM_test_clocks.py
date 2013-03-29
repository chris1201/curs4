__author__ = 'gavr'

from newRTRBM import *
from clocks import *
from utils import *
from tictoc import tic, toc

imageSize = 30
# setCurrentDirectory('26_3_13_rbm_wo_regul_gibbs_step_20_hidden_100_widthline_1_iter_2401_wo_corner')
# clearCurrentDirectory()
setCurrentDirectory('26_3_13_rbm_wo_regul_gibbs_step_10_hidden_500_widthline_1_iter_3201')

SetGreyAsBlack()
# SetDontDrawBlackContour()
dials = DrawDials(Tick(0, 0, 0), Tick(59, 0, 0),  imageSize)
appearance = dials[0]
#   divide to blocks
dataPrime = [convertImageToVector(element) for element in dials]

eachIteration = 200
countStep = 7
countGibbs = 10
learningRate = 0.01

# rbm = OpenRBM(getStringData())
rbm = createSimpleRBM(500, 900)
m = T.matrix()
n = T.iscalar()
s = T.fscalar()
v = T.vector()
print "start create learn function"
grad_func = rbm.grad_function(m, countGibbs, MODE_WITHOUT_COIN, learningRate)
print "start learn"
from_func = rbm.gibbs_function(m, countGibbs, MODE_WITH_COIN_EXCEPT_LAST)
print "Have created function"
rnd_func = rbm.gibbs_function_from_rnd(countGibbs, MODE_WITHOUT_COIN)
print "Have created function"

sample = rbm.bm.generateRandomsFromBinoZeroOne(T.ones_like(rbm.vBias) * 0.5)
res, updates = rbm.bm.gibbs_all(sample, rbm.W, rbm.vBias, rbm.hBias, countGibbs, MODE_WITHOUT_COIN)
rnd_func = theano.function([], res, updates=updates)

tic()
for idx in range(countStep):
    tic()
    print idx, grad_func(dataPrime), toc()
    if idx % eachIteration == 0:
        saveImage(makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(appearance, rnd_func())), "train_step_rnd_1_" + str(idx))
        saveImage(makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(appearance, rnd_func())), "train_step_rnd_2_" + str(idx))
        saveImage(makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(appearance, rnd_func())), "train_step_rnd_3_" + str(idx))
        saveImage(makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(appearance, rnd_func())), "train_step_rnd_4_" + str(idx))

        saveImage(makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(appearance, from_func(dataPrime))), "train_step_from_1_" + str(idx))


print "time learning: ", toc()


saveData(rbm.save())
print "save has been made"
#

saveImage(convertProbabilityVectorToImage(appearance, rnd_func()), "1")
saveImage(convertProbabilityVectorToImage(appearance, rnd_func()), "2")
saveImage(convertProbabilityVectorToImage(appearance, rnd_func()), "3")
saveImage(convertProbabilityVectorToImage(appearance, rnd_func()), "4")
saveImage(convertProbabilityVectorToImage(appearance, rnd_func()), "5")
saveImage(convertProbabilityVectorToImage(appearance, rnd_func()), "6")
saveImage(convertProbabilityVectorToImage(appearance, rnd_func()), "7")


i = T.iscalar()
from_func = rbm.gibbs_function(v, i, MODE_WITHOUT_COIN)
print "Have created function"
for idx in range(1, 15):
    saveImage(convertProbabilityVectorToImage(appearance, from_func(dataPrime[0], idx)), str(idx) + 'from0')

sample = rbm.bm.generateRandomsFromBinoZeroOne(T.ones_like(rbm.vBias) * 0.5)
res, updates = rbm.bm.gibbs_all(sample, rbm.W, rbm.vBias, rbm.hBias, i, MODE_WITHOUT_COIN)
rnd_func = theano.function([i], res, updates=updates)
# rnd_func = rbm.gibbs_function_from_rnd(i, MODE_WITHOUT_COIN)
for idx in range(1, 25):
    saveImage(makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(appearance, rnd_func(idx))), str(idx) + 'from_rnd1')

for idx in range(1, 25):
    saveImage(makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(appearance, rnd_func(idx))), str(idx) + 'from_rnd2')
