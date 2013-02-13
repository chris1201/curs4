from locale import str
import zipfile
import time
from PIL import Image
import StringIO

__author__ = 'gavr'

class ZipManager:
    """
    mode = "a" - updating or create
    mode = "r" - reading
    """
    def __init__(self, filename, mode="a"):
        self.zf = zipfile.ZipFile(filename + '.zip', mode="a")

    def addFile(self, filename):
        try:
            self.zf.write(filename)
        except:
            print 'error adding file into zip-archive'

    def addTextFileFromString(self, filename, text):
        try:
            self.zf.writestr(zipfile.ZipInfo(filename, time.localtime(time.time())[:6]), text)
        except:
            print "Error to write into zip-archive text file"

    def addImage(self, image, filename=None, numGibbsStep=None, indexIteration=None):
        try:
            getCurrentTime = lambda : time.localtime(time.time())[:6]
            transformOrder = lambda t: [t[2], t[1], t[3], t[4], t[5]]
            convertToString = lambda data: '_'.join(map(str, data))
            fn = convertToString(transformOrder(getCurrentTime()))
            if filename is not None:
                fn = fn + filename
            if numGibbsStep is not None:
                fn = fn + '_gibbs_' + str(numGibbsStep)
            if indexIteration is not None:
                fn = fn + '_index_' + str(indexIteration)
            out = StringIO.StringIO()
            image.save(out, format='GIF')
            self.addTextFileFromString(fn + '.gif', out.getvalue())
            out.close()
        except:
            print 'Error in writing Image'

    def __del__(self):
        self.zf.close()

