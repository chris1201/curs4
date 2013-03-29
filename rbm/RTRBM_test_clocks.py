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
app = dials[0];
#   divide to blocks
dataPrime = [convertImageToVector(element) for element in dials];

elementLength = 5;
countStep = 200;
trainBlock = 15;
countGibbs = 5;
learningRate = 0.01
#
#
data = [dataPrime[idx:((idx + elementLength))] + (
    [] if (idx + elementLength) / len(dataPrime) == 0 else dataPrime[:((idx + elementLength) % len(dataPrime))])
                              for idx in range(len(dataPrime))]
#
rtrbm = createSimpleRTRBM(300, imagesize * imagesize)
rtrbm = OpenRTRBM(getStringData())

func = rtrbm.grad_function(10, numpy.asarray(0.01, dtype='float32'), MODE_WITHOUT_COIN)

m = T.matrix()
f, _, u, _, _ = rtrbm.gibbs(m, 1, MODE_WITHOUT_COIN)
f = theano.function([m], f, updates=u)
f5, _, u, _, _ = rtrbm.gibbs(m, 5, MODE_WITHOUT_COIN)
f5 = theano.function([m], f5, updates=u)

for x in u:
    print numpy.shape(x)
#
x1 = numpy.repeat([dataPrime[1]], 5, 0)
x2 = numpy.repeat([dataPrime[5]], 5, 0)

for idx in range(200, 200 + countStep):
    trainBlock = data[1:15]
    tic()
    print idx, func(trainBlock), ', time:', toc()
    if idx % 50 == 0:
        makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(app, f(data[1]))).save(str(idx) + "anim_train6.gif", "GIF")
        makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(app, f(x1))).save(str(idx) + "x1anim_train6.gif", "GIF")
        makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(app, f(x2))).save(str(idx) + "x2anim_train6.gif", "GIF")

        makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(app, f5(data[1]))).save(str(idx) + "anim_train6f5.gif", "GIF")
        makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(app, f5(x1))).save(str(idx) + "x1anim_train6f5.gif", "GIF")
        makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(app, f5(x2))).save(str(idx) + "x2anim_train6f5.gif", "GIF")

makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(app, f(data[1]))).save(str(idx) + "anim_train6.gif", "GIF")
makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(app, f(x1))).save(str(idx) + "x1anim_train6.gif", "GIF")
makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(app, f(x2))).save(str(idx) + "x2anim_train6.gif", "GIF")

makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(app, f5(data[1]))).save(str(idx) + "anim_train6f5.gif", "GIF")
makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(app, f5(x1))).save(str(idx) + "x1anim_train6f5.gif", "GIF")
makeAnimImageFromVectorImages(convertProbabilityMatrixToImages(app, f5(x2))).save(str(idx) + "x2anim_train6f5.gif", "GIF")

saveData(rtrbm.save())
#
# makeAnimImageFromImages(convertProbabilityMatrixToImages(app, f(data[1]))).save("1anim2.gif", "GIF")
# makeAnimImageFromImages(convertProbabilityMatrixToImages(app, f(data[2]))).save("2anim2.gif", "GIF")