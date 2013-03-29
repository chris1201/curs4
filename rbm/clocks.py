from images2gif import writeGif
from PIL import Image, ImageSequence, ImageDraw, ImagePalette
from math import pi, sin, cos

class clockInfo:
    def __init__(self):
        self.hourwidth = 4
        self.minwidth = 2
        self.secwidth = 1
        self.hourlength = 4
        self.isDrawSecond = True
        self.isDrawMinit = False
        self.isDrawHour = False
        #filename = "1.gif"
        self.colorBlack = 0
        self.colorWhite = 1
        self.colorGrey = 2
        self.isDrawBlackCircle = True

info_clock = clockInfo()

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

def DrawDial(ticks, size):

    def div(turple, value):
#    return (x / value for x in turple)
        return (turple[0] / value, turple[1] / value)

    def getPosition(pos, length, sincos):
        return pos + (pos[0] + length * sincos[0], pos[1] + length * sincos[1])

    image = Image.new(mode = "P", size = (size, size))
    image.putpalette([0, 0, 0, 255, 255, 255, 128, 128, 128])
    draw = ImageDraw.Draw(image)
    if info_clock.isDrawBlackCircle:
        draw.ellipse((0, 0) + image.size, fill = info_clock.colorWhite)
    else:
        draw.rectangle((0, 0) + image.size, fill = info_clock.colorWhite)
    minlength = size / 4
    seclength = size / 2
    if (info_clock.isDrawSecond):
        draw.line(getPosition(div(image.size, 2), seclength, ticks.getSecAngle()), fill = info_clock.colorGrey, width = info_clock.secwidth)
    if (info_clock.isDrawMinit):
        draw.line(getPosition(div(image.size, 2), minlength, ticks.getMinAngle()), fill = info_clock.colorGrey, width = info_clock.minwidth)
    if (info_clock.isDrawHour):
        draw.line(div(image.size, 2) + getPosition(info_clock.hourlength, ticks.getHourAngle()), fill = info_clock.colorGrey, width = info_clock.hourwidth)
    del draw
    return image.copy()

def DrawDials(tickbegin, tickend, size):
    tick = Tick(tickbegin.secs, tickbegin.minutes, tickbegin.hours)
    ret = ()
    while (tick != tickend):
        ret = ret + (DrawDial(tick, size),)
        tick.inc()
    del tick
    return ret

def getImagesFromGif(filename):
    im = Image.open(filename)
    return [frame.copy() for frame in ImageSequence.Iterator(im)]

def SetGreyAsBlack():
    info_clock.colorGrey = info_clock.colorBlack

def SwapBlackAndWhite():
    x = info_clock.colorBlack
    y = info_clock.colorWhite
    if info_clock.colorWhite == info_clock.colorGrey:
        info_clock.colorGrey = x

def SetDontDrawBlackContour():
    info_clock.isDrawBlackCircle = False

def SetDrawBlackContour():
    info_clock.isDrawBlackCircle = True


def SetSecWidth(width):
    info_clock.secwidth = width

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