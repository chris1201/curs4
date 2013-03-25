__author__ = 'gavr'

from newRTRBM import *
from clocks import *
from utils import *
from tictoc import tic, toc

imageSize = 30
setCurrentDirectory('25_3_13_rbm_test_temp')
SetGreyAsBlack()
# SetDontDrawBlackContour()
dials = DrawDials(Tick(0, 0, 0), Tick(59, 0, 0), imageSize)
appearance = dials[0]
#   divide to blocks
dataPrime = [convertImageToVector(element) for element in dials]

eachIteration = 50
countStep = 801
countGibbs = 20
learningRate = 0.01


# rbm = OpenRBM(getStringData())
rbm = createSimpleRBM(300, 900)
m = T.matrix()
n = T.iscalar()
s = T.fscalar()
v = T.vector()
print "start create learn function"
grad_func = rbm.grad_function(m, countGibbs, MODE_WITHOUT_COIN, learningRate)
print "start learn"
from_func = rbm.gibbs_function(v, 1, MODE_WITHOUT_COIN)
print "Have created function"
rnd_func = rbm.gibbs_function_from_rnd(1, MODE_WITHOUT_COIN)
print "Have created function"

tic()
for idx in range(countStep):
    tic()
    print idx, grad_func(dataPrime), toc()
    if idx % eachIteration == 0:
        saveImage(convertProbabilityVectorToImage(appearance, rnd_func()), "train_step_rnd_1_" + str(idx))
        saveImage(convertProbabilityVectorToImage(appearance, rnd_func()), "train_step_rnd_2_" + str(idx))
        saveImage(convertProbabilityVectorToImage(appearance, rnd_func()), "train_step_rnd_3_" + str(idx))
        saveImage(convertProbabilityVectorToImage(appearance, rnd_func()), "train_step_rnd_4_" + str(idx))

        saveImage(convertProbabilityVectorToImage(appearance, from_func(dataPrime[0])), "train_step_from_1_" + str(idx))


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

saveImage(convertProbabilityVectorToImage(appearance, from_func(dataPrime[0])), "1f")
saveImage(convertProbabilityVectorToImage(appearance, from_func(dataPrime[1])), "2f")
saveImage(convertProbabilityVectorToImage(appearance, from_func(dataPrime[2])), "3f")
saveImage(convertProbabilityVectorToImage(appearance, from_func(dataPrime[3])), "4f")
saveImage(convertProbabilityVectorToImage(appearance, from_func(dataPrime[4])), "5f")
saveImage(convertProbabilityVectorToImage(appearance, from_func(dataPrime[5])), "6f")
saveImage(convertProbabilityVectorToImage(appearance, from_func(dataPrime[6])), "7f")