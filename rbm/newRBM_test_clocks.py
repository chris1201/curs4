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
countStep = 200;
# trainBlock = 15;
# countGibbs = 5;
# learningRate = 0.01


data = [dataPrime[idx:((idx + elementLength))] + (
    [] if (idx + elementLength) / len(dataPrime) == 0 else dataPrime[:((idx + elementLength) % len(dataPrime))])
                              for idx in range(len(dataPrime))]

rtrbm = createSimpleRTRBM(200, imagesize * imagesize)


func = rtrbm.grad_function(5, numpy.asarray(0.01, dtype='float32'), MODE_WITHOUT_COIN)

m = T.matrix()
f, _, u, _, _ = rtrbm.gibbs(m, 1, MODE_WITHOUT_COIN)
f = theano.function([m], f, updates=u)
# for x in u:
#     print numpy.shape(x)

x1 = numpy.repeat([dataPrime[1]], 5, 0)
x2 = numpy.repeat([dataPrime[5]], 5, 0)

for idx in range(countStep):
    trainBlock = data[1:15]
    tic()
    print idx, func(trainBlock), ', time:', toc()
    if idx % 50 == 0:
        makeAnimImageFromImages(convertProbabilityMatrixToImages(app, f(data[1]))).save(str(idx) + "anim_train5.gif", "GIF")
        makeAnimImageFromImages(convertProbabilityMatrixToImages(app, f(x1))).save(str(idx) + "x1anim_train5.gif", "GIF")
        makeAnimImageFromImages(convertProbabilityMatrixToImages(app, f(x2))).save(str(idx) + "x2anim_train5.gif", "GIF")

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