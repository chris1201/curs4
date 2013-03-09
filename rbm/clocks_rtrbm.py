__author__ = 'gavr'

from clocks import *
from rtrbm import *
from utils import *
from numpy.random.mtrand import shuffle
from tictoc import tic, toc

def train(bm, imagesize, elementLength, countGibbsStep, learningRate, blockTrainImage, countTrainStep):
    #   create data
    dials = DrawDials(Tick(0, 0, 0), Tick(59, 0, 0), imagesize);
    #   divide to blocks
    dataPrime = [convertImageToVector(element) for element in dials];
    data = [dataPrime[idx:(idx + elementLength)]  for idx in range(len(dataPrime))];
    shuffle(data)
    print 'Constructed data'
    print 'Data size: ', len(data)
    tic();
    for idx in range(countTrainStep):
        tic()
        for index in range(0, int(round(len(data) / blockTrainImage))):
            print idx, index, bm.grad_step(data[index * blockTrainImage: (index + 1) * blockTrainImage], countGibbsStep, numpy.asarray(learningRate, dtype='float32'));
        print 'Iteration idx= ', idx , ', time calc = ', toc()
    print 'Learning Time: ', toc();
    return bm

imagesize = 30;
tic();
isDrawMinit = False
colorGrey = colorBlack
print 'Start construct RTRBM'
bm = createSimpleRTRBM(500, imagesize * imagesize);
print 'RTRBM has been constructed, with time = ', toc()
train(bm, imagesize, 3, 3, 0.01, 1, 100)
print 'Training has been ended'
saveData(bm.save().getvalue())
