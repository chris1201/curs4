__author__ = 'gavr'

from clocks import *
from rbm import *
from utils import *
from numpy.random.mtrand import shuffle
from tictoc import tic, toc
import numpy

imagesize = 30;
SetGreyAsBlack()
SetDontDrawBlackContour()
dials = DrawDials(Tick(0, 0, 0), Tick(59, 0, 0), imagesize);
dials[0].save("test.gif", "GIF")
app = dials[0];
#   divide to blocks
dataPrime = [convertImageToVector(element) for element in dials];

# rbm = createSimpleRBM(900, imagesize * imagesize)
rbm = openRBM(getStringData())
# for idx in range(500):
#     tic();
#     print idx, rbm.grad_step(dataPrime, numpy.asarray(0.01, dtype='float32'), 20), toc()

# saveData(rbm.saveTo().getvalue())

# convertVectorToImage(app, rbm.gibbsFromRnd(1)).save("1.gif", "GIF")
# convertVectorToImage(app, rbm.gibbsFromRnd(5)).save("2.gif", "GIF")
# convertVectorToImage(app, rbm.gibbsFromRnd(10)).save("3.gif", "GIF")
# convertVectorToImage(app, rbm.gibbsFromRnd(20)).save("4.gif", "GIF")
# convertVectorToImage(app, rbm.gibbs(dataPrime[0], 1)).save("5.gif", "GIF")
# convertVectorToImage(app, rbm.gibbs(dataPrime[0], 5)).save("6.gif", "GIF")
# convertVectorToImage(app, rbm.gibbs(dataPrime[0], 10)).save("7.gif", "GIF")
# convertVectorToImage(app, rbm.gibbs(dataPrime[0], 20)).save("8.gif", "GIF")

convertVectorToImage(app, rbm.gibbs(dataPrime[0], 20)).save("1.gif", "GIF")
convertVectorToImage(app, rbm.gibbs(dataPrime[1], 20)).save("2.gif", "GIF")
convertVectorToImage(app, rbm.gibbs(dataPrime[2], 20)).save("3.gif", "GIF")
convertVectorToImage(app, rbm.gibbs(dataPrime[3], 20)).save("4.gif", "GIF")
convertVectorToImage(app, rbm.gibbs(dataPrime[4], 20)).save("5.gif", "GIF")
convertVectorToImage(app, rbm.gibbs(dataPrime[5], 20)).save("6.gif", "GIF")
convertVectorToImage(app, rbm.gibbs(dataPrime[6], 20)).save("7.gif", "GIF")

convertVectorToImage(app, rbm.gibbsFromRnd(20)).save("1rnd.gif", "GIF")
convertVectorToImage(app, rbm.gibbsFromRnd(20)).save("2rnd.gif", "GIF")
convertVectorToImage(app, rbm.gibbsFromRnd(20)).save("3rnd.gif", "GIF")
convertVectorToImage(app, rbm.gibbsFromRnd(20)).save("4rnd.gif", "GIF")

