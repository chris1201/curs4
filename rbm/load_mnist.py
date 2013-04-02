__author__ = 'gavr'

from utils import *
from PIL import Image
from rbmTest import *
import os
from StringIO import StringIO

def read_data():
    # for x in os.listdir(os.getcwd() + '/' + ccd.currentDirectory):
    #     print x
    #     Image.open(os.getcwd() + '/' + ccd.currentDirectory + x)
    return [convertImageToVector(Image.open(os.getcwd() + '/' + ccd.currentDirectory + x)) * 1.0 / 255  for x in os.listdir(os.getcwd() + '/' + ccd.currentDirectory)]

def read_appereance():
    return Image.open(os.getcwd() + '/' + ccd.currentDirectory + '1.gif')

setCurrentDirectory('mnist/1')
x = read_data()
setCurrentDirectory('mnist/2')
y = read_data()
z = x + y

app = read_appereance()

string = StringIO()
string.write('mnist2')
string.write('IS_'+str(28))
string.write('_bg_'+str(False))
string.write('_ci_'+str(3200))
string.write('_cg_'+str(5))
string.write('_lr_'+str(0.01))
string.write('_lm_'+MODE_NAMES[MODE_WITHOUT_COIN])
string.write('_h_'+str(50))
string.write('_sW_'+str(0))
string.write('_r_'+str(0))
setCurrentDirectory(string.getvalue())


rbmStohasticGradientTest(outputEveryIteration=5, countGibbs=5, hidden=50, data=z, appearance=app, trainBlock=500)
