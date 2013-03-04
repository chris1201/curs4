__author__ = 'gavr'

import StringIO
import numpy
from PIL import Image

def convertImageToVector(image):
    return numpy.asarray(list(image.getdata()))

def convertVectorToImage(appearance, vector):
    im = appearance.copy()
    im.putdata(vector)
    return im

def convertProbabilityVectorToImage(appereance, vector):
    im = Image.new(mode='F', size=appereance.size)
    im.putdata(map(lambda x: 256 * x, vector))
    return im

def convertMatrixToImages(appearance, matrix):
    return map(lambda x: convertVectorToImage(appearance, x), matrix)

def convertProbabilityMatrixToImages(appearance, matrix):
    return map(lambda x: convertProbabilityVectorToImage(appearance, x), matrix)

# save Data
def saveData(strio):
    file = open('data.txt', 'w')
    file.write(strio)
    file.close()

# readData from data.txt
def getStringData():
    file = open('data.txt', 'r')
    s = StringIO.StringIO()
    output = file.readlines()
    s.writelines(output)
    file.close()
    return s.getvalue()

  