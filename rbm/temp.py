__author__ = 'gavr'

from newRTRBM import *
import numpy as np
import theano.tensor as T
from utils import *
from PIL import Image, ImageDraw

# rtrbm = createSimpleRTRBM(90, 900)
#
# m = T.matrix()
# f, _, u, _, _ = rtrbm.gibbs(m, 1, MODE_WITHOUT_COIN)
# f = theano.function([m], f, updates=u)
# print f(np.zeros(shape=(10, 900)))
# m = T.tensor3()
# f, _, u, _, _ = rtrbm.gibbs(m, 1, MODE_WITHOUT_COIN)
# f = theano.function([m], f, updates=u)
# print f(np.zeros(shape=(10, 10, 900)))
# print rtrbm.grad_function(5, 0.01, MODE_WITHOUT_COIN)(np.zeros(shape=(10, 15, 900)))
# #print numpy.shape(rtrbm.predict_function()
# rtrbm.predict(T.tensor3(), 5, 5, 1)
# q = rtrbm.predict_function(False, 5, 7, 0)
# q1 = rtrbm.predict_function(True, 5, 7, 0)
#
# print numpy.shape(q(numpy.zeros(shape=(10, 900))))
# print numpy.shape(q1(numpy.zeros(shape=(11, 10, 900))))


dataPrime = []

for idx in range(10):
    image = Image.new(mode = "P", size = (10, 10))
    image.putpalette([0, 0, 0, 255, 255, 255])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0) + image.size, fill = 1)
    image.putpixel((idx, idx), 0)
    dataPrime.append(convertImageToVector(image))

app = image

saveOutput = lambda x, name: \
    saveImage( \
        makeAnimImageFromMatrixImages( \
            convertProbabilityTensorToImages(app, x)),
        name)

elementLength = 3

data = [dataPrime[idx:((idx + elementLength))] + (
    [] if (idx + elementLength) / len(dataPrime) == 0 else dataPrime[:((idx + elementLength) % len(dataPrime))])
        for idx in range(len(dataPrime))]


saveOutput(data, 'rtrbm1_data')
