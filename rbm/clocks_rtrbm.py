__author__ = 'gavr'

from clocks import *
from rtrbm import *
from utils import *
from numpy.random.mtrand import shuffle
from tictoc import tic, toc
import numpy

def train(bm, imagesize, elementLength, blockTrainImage, countTrainStep, learningRate, countGibbs):
    #   create data
    dials = DrawDials(Tick(0, 0, 0), Tick(59, 0, 0), imagesize);
    #   divide to blocks
    dataPrime = [convertImageToVector(element) for element in dials];
    print numpy.shape(dataPrime)
    data = [dataPrime[idx:((idx + elementLength))] + (
            [] if (idx + elementLength) / len(dataPrime) == 0 else dataPrime[:((idx + elementLength) % len(dataPrime))])
            for idx in range(len(dataPrime))]
    print 'Constructed data'
    # shuffle(data)
    tic();
    # for index in range(0, int(round(len(data) / blockTrainImage))):
    index = 0
    for idx in range(countTrainStep):
        trainBlock = data[index*blockTrainImage:(index+1)*blockTrainImage]
        if (index+1) * blockTrainImage > len(data):
            trainBlock.append(data[0:((index+1) * blockTrainImage) % len(data)])
        tic()
        print idx, index, bm.grad_step(trainBlock, countGibbs, numpy.asarray(learningRate, dtype='float32')), ', time:', toc()
    # if (idx % 100 == 0):
    #     t = bm.gibbsSamplingPredictionWithOutCoin(data[0], 3, 1)
    #     t = convertProbabilityMatrixToImages(dials[0], t)
    #     t = makeAnimImageFromImages(t)
    #     t.save(str(idx) + ".gif", "GIF")
    #print 'Iteration idx= ', idx, ', time calc = ', toc()

    print 'Learning Time: ', toc();

    t = bm.gibbsSamplingPredictionWithOutCoin(data[0], 3, 1)
    t = convertProbabilityMatrixToImages(dials[0], t)
    t = makeAnimImageFromImages(t)
    t.save("3.gif", "GIF")

    t = bm.gibbsSamplingPredictionWithOutCoin(data[0], 3, 5)
    t = convertProbabilityMatrixToImages(dials[0], t)
    t = makeAnimImageFromImages(t)
    t.save("4.gif", "GIF")

    t = bm.gibbsSamplingPredictionWithOutCoin(data[0], 3, 10)
    t = convertProbabilityMatrixToImages(dials[0], t)
    t = makeAnimImageFromImages(t)
    t.save("5.gif", "GIF")

    t = bm.gibbsSamplingPredictionWithOutCoin(data[5], 3, 25)
    t = convertProbabilityMatrixToImages(dials[0], t)
    t = makeAnimImageFromImages(t)
    t.save("6.gif", "GIF")

    t = bm.gibbsSamplingPredictionWithOutCoin(data[5], 3, 50)
    t = convertProbabilityMatrixToImages(dials[0], t)
    t = makeAnimImageFromImages(t)
    t.save("7.gif", "GIF")

    return bm


tic()
tic()
bm = createSimpleRTRBM(500, 30 * 30)
# bm = openRTRBM(getStringData())
print 'time create RTRBM', toc()
imagesize = 30;
tic();
SetGreyAsBlack()
train(bm, imagesize, 20, 1, 50, 0.01, 10)
print 'Training has been ended: ', toc()
saveData(bm.save().getvalue())
print 'Global time', toc()