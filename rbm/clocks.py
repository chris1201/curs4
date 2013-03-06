from images2gif import writeGif
from PIL import Image, ImageSequence, ImageDraw, ImagePalette
from math import pi, sin, cos

width = 80
height = 80
hourwidth = 4
minwidth = 2
secwidth = 1
hourlength = 8
minlength = 3 * width / 16
seclength = width / 2
filename = "1.gif"

class Tick:
    def __init__(self, secs, minutes, hours):
        self.secs = secs
        self.minutes = minutes
        self.hours = hours
    def inc(self):
        self.secs = (self.secs + 1) % 60
        self.minutes = (self.minutes + (1 if (self.secs == 0) else 0)) % 60
        self.hours = (self.hours + (1 if ((self.secs == 0) and (self.minutes == 0)) else 0)) % 12
    def calcAngle(self, value): return (pi / 30.0) * value - (pi / 2)
    def makeTuple(self, angle): return (cos(angle), sin(angle))
    def getSecAngle(self):  
        return self.makeTuple(self.calcAngle(self.secs))
    def getMinAngle(self):
        return self.makeTuple(self.calcAngle(self.minutes))
    def getHourAngle(self):
        return self.makeTuple(self.calcAngle(self.hours))
    def __ne__(self, other):
        if (isinstance(other, Tick)):
            return not ((self.hours == other.hours) and (self.minutes == other.minutes) and (self.secs == other.secs))
        else:
            return False

def DrawDial(ticks):

    def div(turple, value):
#    return (x / value for x in turple)
        return (turple[0] / value, turple[1] / value)

    def getPosition(pos, length, sincos):
        return pos + (pos[0] + length * sincos[0], pos[1] + length * sincos[1])

    image = Image.new(mode = "P", size = (width, height))
    image.putpalette([0, 0, 0, 128, 128, 128, 255, 255, 255])
    draw = ImageDraw.Draw(image)
    draw.ellipse((0, 0) + image.size, fill = 2)
    draw.line(getPosition(div(image.size, 2), seclength, ticks.getSecAngle()), fill = 1, width = secwidth)
    draw.line(getPosition(div(image.size, 2), seclength, ticks.getMinAngle()), fill = 1, width = minwidth)
    #draw.line(div(image.size, 2) + getPosition(hourlength, ticks.getHourAngle()), fill = 1, width = hourwidth)
    del draw
    return image.copy()

def DrawDials(tickbegin, tickend):
    tick = Tick(tickbegin.secs, tickbegin.minutes, tickbegin.hours)
    ret = ()
    while (tick != tickend):
        ret = ret + (DrawDial(tick),)
        tick.inc()
    del tick
    return ret

def getImagesFromGif(filename):
    im = Image.open(filename)
    return [frame.copy() for frame in ImageSequence.Iterator(im)]

if __name__ == '__main__':
    import time
    #main
    t = time.localtime(time.time())[:6]
    t1 = [t[2], t[1], t[3], t[4], t[5]]
    print t
    print t1
    print '_'.join(map(str, t1))
    getCurrentTime = lambda : time.localtime(time.time())[:6]
    transformOrder = lambda t: [t[2], t[1], t[3], t[4], t[5]]
    convertToString = lambda data: '_'.join(map(str, data))
    getCurrentTimeInMyFormat = convertToString(transformOrder(getCurrentTime()))
    print getCurrentTimeInMyFormat
    #writeGif(filename, DrawDials(Tick(0, 0, 0), Tick(15, 5, 0)), duration = 0.000000001, dither = 1)