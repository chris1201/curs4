__author__ = 'gavr'

from rbmTest import *
from StringIO import StringIO
from tictoc import tic, toc
from PIL import Image

ci = 30001
cg = 10
lr = 0.01
tb = 10
h1 = 7
h = h1 * h1
new_reg = 0.1
reg = 0
lm = MODE_WITHOUT_COIN
rtrbm_ci = 30001
elementLength = 2
regregreg = 1
stoh = 1
# function_make_dir(30, True, ci, cg, lr, lm, h, 1, reg, tb, new_reg)
setCurrentDirectory('TRUEci' + str(ci)+'cg'+str(cg)+'lr'+str(lr)+'tb'+\
    str(tb) + 'h' + str(h) + 'nr' + str(new_reg) + 'lm' + 'woc' + 'rt' + str(rtrbm_ci) + 'nnr' + str(regregreg) + 'stoh' + str(stoh))

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

data = [dataPrime[idx:((idx + elementLength))] + (
    [] if (idx + elementLength) / len(dataPrime) == 0 else dataPrime[:((idx + elementLength) % len(dataPrime))])
        for idx in range(len(dataPrime))]

#
# rtrbm = OpenRTRBM(getStringData('rtrbm20000.txt'))
# saveImage(createFromWeightsImage(theano.function([], rtrbm.W.T)(), 8, 8, (10, 10)), 'W_')
# saveImage(createFromWeightsImage(theano.function([], rtrbm.W.T + rtrbm.vBiasbase)(), 8, 8, (10, 10)), 'W_bias')
# saveImage(createFromWeightsImage(theano.function([], rtrbm.W)(), 10, 10, (8, 8)), 'WT')
# saveImage(createFromWeightsImage(theano.function([], rtrbm.W + rtrbm.hBiasbase)(), 10, 10, (8, 8)), 'WT_')
#
# exit()
# func = rtrbm.predict_function(True, 20, 10, MODE_WITHOUT_COIN)
# saveOutput(func(data), '1')
# elementLength = 1
# data = [dataPrime[idx:((idx + elementLength))] + (
#     [] if (idx + elementLength) / len(dataPrime) == 0 else dataPrime[:((idx + elementLength) % len(dataPrime))])
#         for idx in range(len(dataPrime))]
# saveOutput(func(data), '2')
#
# exit()

rbm=rbmStohasticGradientTest(countGibbs = cg,
                             outputEveryIteration = 2000,
                             countIteration=ci,
                             data=dataPrime,
                             appearance=app,
                             trainBlock=tb,
                             hidden=h,
                             learningMode=lm,
                             regularization=reg,
                             newReg=new_reg,
                             learningRate=lr)
saveImage(createFromWeightsImage(theano.function([], rbm.W)(), 10, 10, (h1, h1)), 'W')
saveImage(createFromWeightsImage(theano.function([], rbm.W.T)(), h1, h1, (10, 10)), 'Wstart')

dataPrime = []

for idx in range(10):
    image = Image.new(mode = "P", size = (10, 10))
    image.putpalette([0, 0, 0, 255, 255, 255])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0) + image.size, fill = 1)
    image.putpixel((idx, idx), 0)
    dataPrime.append(convertImageToVector(image))

app = image

#rbm = OpenRBM(getStringData())
rtrbm = createSimpleRTRBM(h, 100)
rtrbm.vBiasbase = rbm.vBias
rtrbm.hBiasbase = rbm.hBias
rtrbm.W = rbm.W

func = rtrbm.grad_function(1, 0.01, lm, regregreg)

print 'construct_data'

data = [dataPrime[idx:((idx + elementLength))] + (
    [] if (idx + elementLength) / len(dataPrime) == 0 else dataPrime[:((idx + elementLength) % len(dataPrime))])
        for idx in range(len(dataPrime))]
print numpy.shape(data)

funcSample1 = rtrbm.predict_function(True, 2, 1, MODE_WITHOUT_COIN)
funcSample5 = rtrbm.predict_function(True, 2, 5, MODE_WITHOUT_COIN)
funcSample6 = rtrbm.predict_function(True, 2, 10, MODE_WITHOUT_COIN)


#saveOutput(funcSample5(data), 'rtrbm0')
saveOutput(data, 'rtrbm_data')
tic()
tic()
for iter in range(rtrbm_ci):
    if stoh:
        for inner_iter in range((len(data))):
            x = func([data[inner_iter]])
    else:
        x = func(data)
    if (iter % 1000 == 0):
        print 'output, x:', x, 'time', toc()
        tic()
        saveOutput(funcSample1(data), 'rtrbm_output1' + str(iter))
        saveOutput(funcSample5(data), 'rtrbm_output5' + str(iter))
        saveOutput(funcSample6(data), 'rtrbm_output6' + str(iter))
        saveData(rtrbm.save(), 'rtrbm' + str(iter) + '.txt')
toc()
print 'time', toc()

# saveImage(createFromWeightsImage(theano.function([], rtrbm.W.T)(), 6, 6, (10, 10)), 'W1')

# print 'start part 2'

# func = rtrbm.grad_function(10, 0.01, lm, 0.01)

# print 'start train in part 2'
# tic()
# tic()
# for iter in range(rtrbm_ci):
#    for inner_iter in range((len(data))):
#        x = func([data[inner_iter]])
#     x = func(data)
#     if (iter % 1000 == 0):
#         print 'output, x:', x, 'time', toc()
#         tic()
#         saveOutput(funcSample1(data), 'part_2rtrbm_output1' + str(iter))
#         saveOutput(funcSample5(data), 'part_2rtrbm_output5' + str(iter))
#         saveOutput(funcSample6(data), 'part_2rtrbm_output6' + str(iter))
#         saveData(rtrbm.save(), 'part_2rtrbm' + str(iter) + '.txt')
# toc()
# print 'time', toc()

# saveImage(createFromWeightsImage(theano.function([], rtrbm.W.T)(), 8, 8, (10, 10)), 'W')
