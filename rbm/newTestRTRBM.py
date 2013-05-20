from newRTRBM import *
from rbmTest import *
import matplotlib.pyplot as plt

def test():
    rtrbm = createSimpleRTRBM(100, 900)
    
    lm = MODE_WITHOUT_COIN 
    funcSample = rtrbm.predict_function(True, 5, 1, lm)
    funcSample1 = rtrbm.predict_function(True, 5, 5, lm)
    funcSample2 = rtrbm.predict_function(True, 5, 10, lm)
    
    func2 = rtrbm.grad2_function(0.001)
    func = rtrbm.grad_function(1, 0.01, lm, 0)
    
    app, dataPrime = rbmGenerateClocks(30)
    
    elementLength = 2
    data = [dataPrime[idx:((idx + elementLength))] + (
        [] if (idx + elementLength) / len(dataPrime) == 0 else dataPrime[:((idx + elementLength) % len(dataPrime))])
            for idx in range(len(dataPrime))]
    
    res2 = []
    res1 = []
    for i in range(20):
        for index in range(200):
            temp = func2(data)
            res2.append(temp)
        print temp
        for index in range(200):
            temp = func(data)
            res1.append(temp)
        print temp

    saveOutput = lambda x: makeAnimImageFromMatrixImages( convertProbabilityTensorToImages(app, x))
    saveOutput(funcSample(data)).show()
    saveOutput(funcSample1(data)).show()
    saveOutput(funcSample2(data)).show()
    plt.clf()
    plt.cla()
    plt.subplot(121)
    plt.plot(res1, 'r')
    plt.subplot(122)
    plt.plot(res2)
    plt.show()
