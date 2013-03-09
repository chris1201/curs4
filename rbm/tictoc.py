__author__ = 'gavr'

import time

class Stack:
    def __init__(self):
        self.__stack__ = [];
    def push(self, element):
        self.__stack__.append(element);
    def pop(self):
        if not self.isEmpty():
            return self.__stack__.pop()
    def isEmpty(self):
        return len(self.__stack__) == 0;

StackTicToc = Stack();

def tic():
    StackTicToc.push(time.time())

def toc():
    return (-StackTicToc.pop() + time.time())
