__author__ = 'gavr'

import StringIO
import numpy
import os
import shutil
import Image, ImageDraw

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

def convertTensorToImages(appearance, tensor):
    return map(lambda x: convertMatrixToImages(appearance, x), tensor)

def convertProbabilityMatrixToImages(appearance, matrix):
    return map(lambda x: convertProbabilityVectorToImage(appearance, x), matrix)

def convertProbabilityTensorToImages(appearance, tensor):
    return map(lambda x: convertProbabilityMatrixToImages(appearance, x), tensor)

# save Data
def saveData(strio, filename = 'data.txt'):
    file = open(ccd.currentDirectory + filename, 'w')
    file.write(strio)
    file.close()

# readData from data.txt
def getStringData(name='data.txt'):
    file = open(ccd.currentDirectory + name, 'r')
    s = StringIO.StringIO()
    output = file.readlines()
    s.writelines(output)
    file.close()
    return s.getvalue()

def makeAnimImageFromVectorImages(data):
    count = len(data)
    size0 = data[0].size
    size = (size0[0], count * size0[1])
    imag = Image.new(size=size, mode=data[0].mode)
    if imag.mode == 'P':
        imag.putpalette(data[0].getpalette())
    for idx in range(0, count):
        imag.paste(data[idx], (0, idx * size0[1], size0[1], (idx + 1) * size0[1]))
    return imag

def makeAnimImageFromMatrixImages(data):
    count1 = len(data)
    count2 = len(data[0])
    size0 = data[0][0].size
    size = (count2 * size0[0] + count2 - 1, count1 * size0[1] + count1)
    imag = Image.new(size=size, mode=data[0][0].mode)
    if imag.mode == 'P':
        imag.putpalette(data[0][0].getpalette())
    for idx1 in range(count1):
        for idx2 in range(count2):
            imag.paste(data[idx1][idx2], \
                (idx2 * size0[0] + idx2, idx1 * size0[1] + idx1 + 1, \
                 (idx2 + 1) * size0[0] + idx2, (idx1 + 1) * size0[1] + idx1 + 1))
    dr = ImageDraw.Draw(imag)
#    for idx1 in range(count1 - 1):
    for idx1 in range(count1):
        dr.line((0, idx1 * size0[1] + idx1, size[0], idx1 * size0[1] + idx1), 128)
    for idx2 in range(1, count2):
        dr.line((idx2 * size0[0] + idx2 - 1, 0, idx2 * size0[0] + idx2 - 1, size[1]), 128)
    return imag

def saveImage(image, filename, ext='GIF'):
    image.save(ccd.currentDirectory + filename + '.' + ext, ext)

def createFromWeightsImages(W, h1, h2, imageSize):
    max = numpy.max(W)
    min = numpy.min(W)
    W = map(lambda x: map(lambda y: (y - min) / (max - min), x), W)
    output = []
    app = Image.new(mode='P', size=imageSize)
    for idx1 in range(h1):
        output.append([])
        for idx2 in range(h2):
            output[idx1].append(convertProbabilityVectorToImage(app, W[idx1 * h2 + idx2]))
    return output

def createFromWeightsImage(W, h1, h2, imageSize):
    return makeAnimImageFromMatrixImages(createFromWeightsImages(W, h1, h2, imageSize))

class ContainerCurrentDirectory:
    def __init__(self):
        self.currentDirectory = ''

ccd = ContainerCurrentDirectory()

def setCurrentDirectory(name):
    ccd.currentDirectory = name + '/'
    print "set current dir: ", ccd.currentDirectory
    if not os.path.exists(ccd.currentDirectory):
        os.makedirs(ccd.currentDirectory)

def clearCurrentDirectory():
    shutil.rmtree(ccd.currentDirectory)
# setCurrentDirectory('26_3_13_rbm_wo_regul_gibbs_step_20_hidden_100_widthline_1_iter_2401')
# setCurrentDirectory('26_3_13_rbm_wo_regul_gibbs_step_80_hidden_100_widthline_1_iter_2401')
# setCurrentDirectory('26_3_13_rbm_wo_regul_gibbs_step_40_hidden_200_widthline_1_iter_2401')

# setCurrentDirectory('26_3_13_rbm_wo_regul_gibbs_step_10_hidden_300_widthline_2_iter_2401')
# clearCurrentDirectory()

# todo image concatinate by horizontal, by vertical
# todo plot full image(ala mesh)
#
