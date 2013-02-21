import utils

__author__ = 'gavr'

from rbm import RBM
from rbm import createSimpleRBM
from utils import convertImageToVector
from utils import convertVectorToImage
from utils import saveData
from PIL import Image
from PIL import ImageDraw
from numpy.oldnumeric.random_array import random_integers
import numpy

def generatorImage(size):
    image = Image.new(mode = "P", size = (size, size))
    image.putpalette([255, 255, 255, 0, 0, 0])
    draw = ImageDraw.Draw(image)
    f = lambda x, y: random_integers(y, minimum=x)
    draw.line((f(1, size/2), f(1, size/2), f(size/2, size), f(size/2, size)), fill = 1)
    return image

def generatorWrongImage(size):
    image = Image.new(mode = "P", size = (size, size))
    image.putpalette([255, 255, 255, 0, 0, 0])
    draw = ImageDraw.Draw(image)
    f = lambda x, y: random_integers(y, minimum=x)
    draw.line((f(size / 2, size), f(1, size / 2), f(1, size / 2), f(size / 2, size)), fill = 1)
    return image

size = 20
# generate Data
datasize = 2000
data = [convertImageToVector(generatorImage(size)) for i in range(0, datasize)]
rbm = createSimpleRBM(100, size * size)
#saveData(rbm.saveTo().getvalue())
#rbm = openRBM(getStringData())
print 'start train'

for idx in range(0, 80):
    for index in range(0, 20):
        print idx, rbm.grad_step(data[index * 100: (index+1) * 100 - 1], numpy.asarray(0.01, dtype='float32'), 20)

print 'control train data'

for obj in data:
    print rbm.freeEnergy(obj)

print 'control train data'

data = [convertImageToVector(generatorImage(size)) for i in range(0, 10)]

for obj in data:
    print rbm.freeEnergy(obj)

print 'randomInfo'

for idx in range(0, 5):
    x = rbm.generateVisibles()
    print rbm.freeEnergy(x)
    x1 = rbm.gibbs(x, 1)
    print rbm.freeEnergy(x1)
    x2 = rbm.gibbs(x, 10)
    print rbm.freeEnergy(x2)

print 'WringImage'

for idx in range(0, 5):
    x = generatorWrongImage(size)
    x = convertImageToVector(x)
    print rbm.freeEnergy(x)
    x1 = rbm.gibbs(x, 1)
    print rbm.freeEnergy(x1)
    x2 = rbm.gibbs(x, 10)
    print rbm.freeEnergy(x2)

convertVectorToImage(generatorImage(size), rbm.gibbs(convertImageToVector(generatorImage(size)), 1)).show()
convertVectorToImage(generatorImage(size), rbm.gibbs(convertImageToVector(generatorImage(size)), 5)).show()
convertVectorToImage(generatorImage(size), rbm.gibbs(convertImageToVector(generatorImage(size)), 10)).show()
convertVectorToImage(generatorImage(size), rbm.gibbs(convertImageToVector(generatorImage(size)), 20)).show()
convertVectorToImage(generatorImage(size), rbm.gibbs(convertImageToVector(generatorImage(size)), 30)).show()

convertVectorToImage(generatorImage(size), rbm.gibbsFromRnd(1)).show()
convertVectorToImage(generatorImage(size), rbm.gibbsFromRnd(5)).show()
convertVectorToImage(generatorImage(size), rbm.gibbsFromRnd(10)).show()
convertVectorToImage(generatorImage(size), rbm.gibbsFromRnd(20)).show()
convertVectorToImage(generatorImage(size), rbm.gibbsFromRnd(30)).show()


saveData(rbm.saveTo().getvalue())
print 'saving has been made'