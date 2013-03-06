__author__ = 'gavr'

import rtrbm
from PIL import Image
from PIL import ImageDraw
from numpy.oldnumeric.random_array import random_integers
import utils
import numpy

def generatorImage(size, quadrants):
    image = Image.new(mode = "P", size = (size, size))
    image.putpalette([255, 255, 255, 0, 0, 0])
    draw = ImageDraw.Draw(image)
    f_temp = lambda d: random_integers(d[1], minimum=d[0])
    f = lambda d: map(f_temp, d)
    coords = [1, size/2, size]
    variants = [(coords[0], coords[1]), (coords[1], coords[2])];
    defquadrants = [(variants[0], variants[0]), (variants[0], variants[1]), (variants[1], variants[0]), (variants[1], variants[1])]
    orders = [1, 0, 2, 3]
    print defquadrants
    data = [defquadrants[orders[i-1]] for i in quadrants];
    print data
    data = map(f, data)
    print data
    result = [item for sublist in data for item in sublist]
    #    [(u) for u in data]
#    result = (u for u in result)
    print result
    draw.line(result, fill = 1)
    return image

def generatorAnimImag(size, xy = None):
    image = Image.new(mode = "P", size = (size, size))
    image.putpalette([255, 255, 255, 0, 0, 0])
    draw = ImageDraw.Draw(image)
    if (xy is not None):
        image.putpixel(xy, 1)
    return image

def generatorAnimImages(size, range):
    return [generatorAnimImag(size, (u)) for u in range]

def generatorAnimLine(size, start, count):
    return generatorAnimImages(size, zip(range(start, start+count), range(start, start + count)))

def convertAnimToDataBlock(data):
    return [utils.convertImageToVector(d) for d in data]

def makeAnimImageFromImages(data):
    count = len(data)
    size0 = data[0].size
    size = (size0[0], count * size0[1])
    imag = Image.new(size=size, mode=data[0].mode)
    if imag.mode == 'P':
        imag.putpalette(data[0].getpalette())
    for idx in range(0, count):
        imag.paste(data[idx], (0, idx * size0[1], size0[1], (idx + 1) * size0[1]))
    return imag

def train(bm, data, countStep, countGibbs, learningRate):
    for idx in range(1, countStep):
        print idx, bm.grad_step(data, countGibbs, numpy.asarray(learningRate, dtype='float32'))

def trainByElement(bm, data, countStep, countGibbs, learningRate):
    for idx in range(1, countStep):
        for index in range(0, len(data)):
            print idx, index, bm.grad_step([data[index]], countGibbs, numpy.asarray(learningRate, dtype='float32'))


# initial
size = 10
countFrames = 2
appereance = generatorAnimImag(size)
#data
data = [convertAnimToDataBlock(generatorAnimLine(size, idx, countFrames)) for idx in range(0, 10 - countFrames)]

mode = 1
if mode == 1:
    #new_rbm
    bm = rtrbm.createSimpleRTRBM(200, size * size)
    print 'learning has started'
#    train(bm, data, 200, 1, 0.01)
    trainByElement(bm, data, 50, 5, 0.01)
    utils.saveData(bm.save().getvalue())
    print 'to save was done'
else:
    #load_rbm
    bm = rtrbm.openRTRBM(utils.getStringData())

makeAnimImageFromImages(temp).show()
