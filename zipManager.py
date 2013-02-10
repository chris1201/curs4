import string
import zipfile
import time
from PIL import Image
import StringIO

__author__ = 'gavr'

class ZipManager:
    def __init__(self, filename):
        self.zf = zipfile.ZipFile.open(filename + '.zip', mode="w+")

    def addFile(self, filename):
        try:
            self.zf.write(filename)
        except:
            print 'error adding file into zip-archive'

    def addTextFileFromString(self, filename, text):
        if (text is string):
            try:
                self.zf.writestr(zipfile.ZipInfo(filename, time.localtime(time.time())[:6]), text)
            except:
                print "Error to write into zip-archive"
        else:
            print 'varible text is not string'

    def addImage(self, filename, image):
        if image is Image:
            out = StringIO.StringIO()
            image.save(out)
            self.addTextFile(filename, out.getvalue())
            out.close()
        else:
            print "image is not object of class Image from PIL"

    def __del__(self):
        self.zf.close()

class EmptyZipManager:
    def __init__(self, filename):
        pass
    def addFile(self, filename):
        pass
    def addTextFileFromString(self, filename, text):
        pass
    def addImage(self, filename, image):
        pass




