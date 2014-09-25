import re
import numpy as np

"""
utility functions to get data on elements, though deepGet and deepest can have 
more generic usage
"""

#output of ElementData[#, "ElectronConfiguration"] & /@ Range[100] in 
#mathematica, i.e. electron configurations of the first 100 elements
s = open("../data/configurations.txt", 'r').read()

#remove MMA list delimiters
s2 =s.replace("{","").replace("}", "").replace("\r", "") 

#convert from string representation of an int list of depth 1
toIntList = lambda x: map(int, x.split(",")) 

#list representing electron configurations  indexed by z, n, l
eConfigs = map(lambda x: map(toIntList, x.split("\t")), s2.split('\n'))

#originally part of the mu.py package
#TODO: it's wasteful to open a file every time this is called
def getElementName(atomicN): 
    elementNamesf = '../data/elementNames.csv'
    elementLabels = np.genfromtxt(elementNamesf, dtype = (np.dtype('S10'), np.dtype('S10')))
    extractedLabel = filter(lambda x : atomicN == int(x[0]), elementLabels)
    if len(extractedLabel) == 0: 
        raise ValueError("Element " + element + "not found. Valid keys \
            are: ")
    name = extractedLabel[0][1]
    return name

def deepGet(arr, indexarr):
    """
    for a (not necessarily rectangular) list of arbitrary depth with an existing element
    indexed by x1, x2, x3,... xn, return element corresponding to values of x1, 
    x2, x3,... xm, where m <= n, given by indexarr.
    """
    if len(indexarr) == 0:
        return arr
    elif type(arr) != list:
        raise ValueError("index array too long")
    else:
        return deepGet(arr[indexarr[0]], indexarr[1:])

def deepest(arr):
    """
    find the location of the deepest (i.e. bottom rightmost in an ordered
     tree representation) element of the list.
    """
    if type(arr) != list:
        return []
    else:
        return [len(arr) - 1] + deepest(arr[-1])

class electronConfig:
    """
    stores electron configuration for a single element, provides methods to 
    get population of orbitals

    attributes:
        tree: lst representation of the element's electron configuration
        element: string representation of element's name
        nMax: highest populated principal energy level
    methods:
        electrons: give population of nth energy level, or of orbital n, l
    """

    def __init__(self, atomicN):
        self.tree = eConfigs[atomicN - 1]
        self.element = getElementName(atomicN)
        self.nMax = len(self.tree)

    def electrons(self, n, l = 'all'):
        if type(n) != int or n < 1 or n > self.nMax:
            raise ValueError("invalid energy level: " + str(n))
        if l == 'all':
            return deepGet(self.tree, [n - 1])
        else:
            return deepGet(self.tree, [n - 1, l])

    def printConfig(self):
        """
        return the electron configuration as a string in a format acceptable
        for Robert Cowan's atomic structure code input files. 

        """
        lMap = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
        configStr = ""
#        if outerOnly:
#            subTree = [self.tree[-1]]
#            nRange = [len(self.tree)]
        nRange = range(1, len(self.tree) + 1)
        for n in nRange:
            for l in range(len(self.tree[n - 1])):
                configStr += str(n) + lMap[l] 
                ePop =  self.electrons(n, l)
                if ePop != 1:
                    configStr += str(ePop)
                configStr += " "
        return configStr[:-1]


