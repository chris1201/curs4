__author__ = 'gavr'

from rtrbm import *
from utils import *
from clocks import *
from numpy import shape

SetGreyAsBlack()
imagesize = 30
dials = DrawDials(Tick(0, 0, 0), Tick(59, 0, 0), imagesize);
#   divide to blocks
dataPrime = [convertImageToVector(element) for element in dials];
print numpy.shape(dataPrime)
elementLength = 3
data = [dataPrime[idx:((idx + elementLength))] + (
    [] if (idx + elementLength) / len(dataPrime) == 0 else dataPrime[:((idx + elementLength) % len(dataPrime))])
        for idx in range(len(dataPrime))]

bm = openRTRBM(getStringData())
print "RTRBM was loaded"
res = [bm.gibbsSamplingPrediction(sample, 3, 10) for sample in data]
t = res[0]
t = map(lambda x: convertVectorToImage(dials[0], x), t)
t = makeAnimImageFromImages(t)
t.save("1.gif", "GIF")
dials[0].save("2.gif", "GIF")
# res2 = [map(lambda x: convertVectorToImage(dials[0], x), object) for object in data]
# res3 = map(makeAnimImageFromImages, res2)

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

print "end output"

# for idx in range(len(res3)):
#     res3[idx].save(str(idx) + ".gif", "GIF")
# res3[0].save("1.gif", "GIF")

# res = [bm.gibbsSamplingPredictionWithOutCoin(sample, 3, 5) for sample in data]
# res2 = [map(lambda x: convertProbabilityVectorToImage(dials[0], x), object) for object in data]
# res3 = map(makeAnimImageFromImages, res2)

# for idx in range(len(res3)):
#     res3[idx].save("prob" + str(idx) + ".jpeg", "JPEG")
# res3[0].save("2.gif", "GIF")