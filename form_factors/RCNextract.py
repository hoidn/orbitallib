# coding: utf-8
#extract radial wave functions from Robert Cowan's code
#empirical normalization factor
import numpy as np
import extractWF
import sys
import pandas as pd
import mu
import elementdata
import os
import ipdb

def reorder(arr, stride):
    """
    given that columns of arr consist of stride number of concatenated data 
    columns of the same length (such as rdat), separate the data and transform 
    columns to rows
    """
    startShape = np.shape(arr)
    newShape = (stride * startShape[1], startShape[0]/stride)
    rowMajorInterleaved = np.reshape(arr.T, newShape)
    components = np.array([rowMajorInterleaved[start::stride] for start in range(stride)])
    return np.vstack(components)


Z = int(sys.argv[1]) #atomic number
NUMBLOCKS = int(os.popen('bash countblocks.sh').read()) #blocks of wf data in out36

#bohr/angstom conversion factor
SCALEFAC = np.sqrt(extractWF.BOHRTOA)#np.sqrt(0.5795985060690942)

#masks defining column numbers of radial wave function data for NUMBLOCKS = 0 and 1
#TODO: extend for NUMBLOCKS > 1 (for heavier elements)
dataMask = [range(8, 12), range(8, 12) + range(14, 24)]

#strip superfluous text from out36 and save result to out36_clean
os.system('bash clean_output.sh')
df = pd.read_table('out36_clean', sep='\s+', header=None)
df.fillna(0, inplace=True)
rdat = np.array(df)

NUMROWS = len(rdat)/NUMBLOCKS

reshapedDat = reorder(rdat, NUMBLOCKS)
radii = reshapedDat[0]

real = SCALEFAC * reshapedDat[dataMask[NUMBLOCKS - 1]][::2]
imag = SCALEFAC * reshapedDat[dataMask[NUMBLOCKS - 1]][1::2]

# highest n, l (in that order) valence orbital. Note that elementConfig denotes:
#n = 1, 2, .. as 0, 1, etc. 
#l = 0, 1, .. as 0, 1, etc.
elementConfig = elementdata.electronConfig(Z)
orbitalNum = 0 #to keep track of column location in wave function input file
for n in range(1, elementConfig.nMax + 1):
    #num of electrons in a given orbital
    for nElectrons in elementConfig.electrons(n):
        real[orbitalNum] *= np.sqrt(nElectrons)
        imag[orbitalNum] *= np.sqrt(nElectrons)
        orbitalNum += 1
        

#q points
x = np.linspace(.01, 25, 200)
mt = extractWF.Orbitals(radii, real, imag, str(elementConfig.element))
#TODO: automate this
ffacs = [mt.shellFormFactor(x, n) for n in range(7)]
yffacs = [ff[1] for ff in ffacs]

#TODO: automate this
#through1s = np.sum(yffacs[:4], axis = 0)
#ones, twos, twop, threes = yffacs[:4]
#threep = yffacs[4]
#valence = np.sum(yffacs[5:7], axis = 0)
all = np.sum(yffacs, axis = 0)
#np.savetxt(elementConfig.element + '_atomic_form_factor.txt', zip(x, through1s, threep, valence, threes, all), delimiter='\t', header='q (inverse Anstrom)\tf(q): 1s,2s,2p,3s\tf(q): 3p\tf(q): 4s,3d\tf(q): 3s\tf(q): total')
np.savetxt('../data/' + elementConfig.element + '_atomic_form_factor_by_orbital.txt', np.vstack((x, yffacs, all)).T, delimiter='\t', header='q (inverse Anstrom)\tf(q): 1s\tf(q): 2s\tf(q): 2p\tf(q): 3s\tf(q): 3p\tf(q): 3d\tf(q): 4s\tf(q): total')

