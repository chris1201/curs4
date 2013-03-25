__author__ = 'gavr'

from newRTRBM import *
from clocks import *
from utils import *
from tictoc import tic, toc

imagesize = 30;
SetGreyAsBlack()
# SetDontDrawBlackContour()
# imagesize
dials = DrawDials(Tick(0, 0, 0), Tick(59, 0, 0), imagesize);
dials[0].save("test.gif", "GIF")
app = dials[0];
#   divide to blocks
dataPrime = [convertImageToVector(element) for element in dials];

elementLength = 5;
countStep = 50;
trainBlock = 15;
countGibbs = 5;
learningRate = 0.01


data = [dataPrime[idx:((idx + elementLength))] + (
    [] if (idx + elementLength) / len(dataPrime) == 0 else dataPrime[:((idx + elementLength) % len(dataPrime))])
                              for idx in range(len(dataPrime))]

rtrbm = createSimpleRTRBM(3000, imagesize * imagesize)

# func = rtrbm.grad_function(countGibbs, numpy.asarray(learningRate, dtype='float32'), MODE_WITHOUT_COIN)

# tic()

# print func(data[1:15]), "   ololo   ",toc()

m = T.matrix()
count = T.iscalar()
f, a, u, b, c = rtrbm.gibbs(m, 2, MODE_WITHOUT_COIN)
f = theano.function([m], [f, a, b, c], updates=u)
for x in f(numpy.zeros((10, 900))):
    print numpy.shape(x)

# for x in f(numpy.zeros((10, 900)), 2):
#     print numpy.shape(x)


FUNCTION_MODE = MODE_WITHOUT_COIN
countGibbs = 2
m = T.tensor3()
energy, gradVarible, gradient, updates = rtrbm.gradient(m, countGibbs, FUNCTION_MODE)
rtrbm.bm.addGradientToUpdate(updates, gradVarible, gradient, numpy.asarray(0.01, dtype='float32'))
func = theano.function([m], energy, updates=updates)
tic()
u = func(data[1:15])
print u, toc()
# for x in u:
#     print numpy.shape(x)

# for idx in range(countStep):
#     trainBlock = data[1:15]
#     tic()
#     print idx, func(trainBlock), ', time:', toc()
    # if idx % 20 == 0:
    #     makeAnimImageFromImages(convertProbabilityMatrixToImages(app, f(data[1]))).save(str(idx) + "anim_train2.gif", "GIF")

# saveData(rtrbm.save())

# makeAnimImageFromImages(convertProbabilityMatrixToImages(app, f(data[1]))).save("1anim2.gif", "GIF")
# makeAnimImageFromImages(convertProbabilityMatrixToImages(app, f(data[2]))).save("2anim2.gif", "GIF")



# rbm = OpenRBM(getStringData())
# rbm = createSimpleRBM(1000, 900)
# m = T.matrix()
# n = T.iscalar()
# s = T.fscalar()
# v = T.vector()
# print "start create learn function"
# grad_func = rbm.grad_function(m, 20, MODE_WITHOUT_COIN, 0.01)
# print "start learn"
# tic()
# for idx in range(600):
#     tic()
#     print idx, grad_func(dataPrime), toc()#, 20, numpy.asarray(0.01, dtype='float32')), toc()
# print "time learning: ", toc()
#
# saveData(rbm.save())
#
# rnd_func = rbm.gibbs_function_from_rnd(1, MODE_WITHOUT_COIN)
# print "Have created function"
# convertProbabilityVectorToImage(app, rnd_func()).save("1.gif", "GIF")
# convertProbabilityVectorToImage(app, rnd_func()).save("2.gif", "GIF")
# convertProbabilityVectorToImage(app, rnd_func()).save("3.gif", "GIF")
# convertProbabilityVectorToImage(app, rnd_func()).save("4.gif", "GIF")
# convertProbabilityVectorToImage(app, rnd_func()).save("5.gif", "GIF")
# convertProbabilityVectorToImage(app, rnd_func()).save("6.gif", "GIF")
# convertProbabilityVectorToImage(app, rnd_func()).save("7.gif", "GIF")
#
# func = rbm.gibbs_function(v, 1, MODE_WITHOUT_COIN)
# print "Have created function"
# convertProbabilityVectorToImage(app, func(dataPrime[0])).save("1f.gif", "GIF")
# convertProbabilityVectorToImage(app, func(dataPrime[1])).save("2f.gif", "GIF")
# convertProbabilityVectorToImage(app, func(dataPrime[2])).save("3f.gif", "GIF")
# convertProbabilityVectorToImage(app, func(dataPrime[3])).save("4f.gif", "GIF")
# convertProbabilityVectorToImage(app, func(dataPrime[4])).save("5f.gif", "GIF")
# convertProbabilityVectorToImage(app, func(dataPrime[5])).save("6f.gif", "GIF")
# convertProbabilityVectorToImage(app, func(dataPrime[6])).save("7f.gif", "GIF")