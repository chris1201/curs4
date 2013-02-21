__author__ = 'gavr'

import StringIO
import numpy

def convertImageToVector(image):
    return numpy.asarray(list(image.getdata()))

def convertVectorToImage(appearance, vector):
    im = appearance.copy()
    im.putdata(vector)
    return im

# save Data
def saveData(strio):
    file = open('data000.txt', 'w')
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

  