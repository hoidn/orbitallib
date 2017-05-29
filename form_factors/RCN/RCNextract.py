# coding: utf-8
#extract radial wave functions from Robert Cowan's code
#empirical normalization factor
import numpy as np
import extractWF
import pandas as pd
import os
from atomicform import atomicform
import pdb

def reorder(arr, stride):
    """
    given that columns of arr consist of stride number of concatenated data 
    columns of the same length (such as rdat), separate the data and transform 
    columns to rows
    """
    startShape = np.shape(arr)
    newShape = (stride * startShape[1], startShape[0]//stride)
    rowMajorInterleaved = np.reshape(arr.T, newShape)
    components = np.array([rowMajorInterleaved[start::stride] for start in range(stride)])
    return np.vstack(components)

def import_WF(data_directory, Z):
    """
    Returns radii, real, Z
    """
    #NUMBLOCKS = int(os.popen('bash %s/countblocks.sh' % data_directory).read()) #blocks of wf data in out36
    NUMBLOCKS = 2 # temporary for MgO calculation
    #ELEMENT = mu.getElementName(Z) 

    #bohr/angstom conversion factor
    # TODO why the square root?
    SCALEFAC = np.sqrt(extractWF.BOHRTOA)#np.sqrt(0.5795985060690942)

    #masks defining column numbers of radial wave function data for NUMBLOCKS = 0 and 1
    dataMask = [list(range(8, 12)), list(range(8, 12)) + list(range(14, 24))]

    #strip superfluous text from out36 and save result to out36_clean
    #os.system('bash %s/clean_output.sh' % data_directory)
    df = pd.read_table(data_directory + '/out36_clean', sep='\s+', header=None)
    df.fillna(0, inplace=True)
    rdat = np.array(df)

    reshapedDat = reorder(rdat, NUMBLOCKS)
    #radii = reshapedDat[0] * extractWF.BOHRTOA
    radii = reshapedDat[0] * SCALEFAC
    #elementConfig = atomicform.eConfigs[Z - 1] #electron configuration for this particular element

    rRnl = reshapedDat[dataMask[NUMBLOCKS - 1]][::2][:atomicform.nOrbitals[Z - 1]]
    return radii, rRnl



#
##q points
#x = np.linspace(.01, 25, 200)
#mt = extractWF.Orbitals(radii, real, imag, str(ELEMENT))
##TODO: automate this
#ffacs = [mt.shellFormFactor(x, n) for n in range(7)]
#yffacs = [ff[1] for ff in ffacs]
#
##TODO: automate this
#all = np.sum(yffacs, axis = 0)
#np.savetxt(ELEMENT + '_atomic_form_factor_by_orbital.txt', np.vstack((x, yffacs, all)).T, delimiter='\t', header='q (inverse Anstrom)\tf(q): 1s\tf(q): 2s\tf(q): 2p\tf(q): 3s\tf(q): 3p\tf(q): 3d\tf(q): 4s\tf(q): total')
#
