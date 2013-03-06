__author__ = 'gavr'

from clocks import *
from rtrbm import *
from utils import *
from numpy.random.mtrand import shuffle

def train(bm, elementLength, countGibbsStep, learningRate, blockTrainImage, countTrainStep):
    #   create data
    dials = DrawDials(Tick(0, 0, 0), Tick(59, 59, 11));
    #   divide to blocks
    dataPrime = [convertImageToVector(element) for element in dials];
    data = [dataPrime[idx:(idx + elementLength)]  for idx in range(len(dataPrime))];
    shuffle(data)
    for idx in range(countTrainStep):
        for index in range(0, round(len(data) / blockTrainImage)):
            print idx, index, bm.grad_step([data[index * blockTrainImage: (index + 1) * blockTrainImage]], countGibbsStep, numpy.asarray(learningRate, dtype='float32'));
    return bm

