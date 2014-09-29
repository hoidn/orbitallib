import ipdb
from numpy import fft
import numpy as np

#length conversion factor
BOHRTOA = 0.52918
TINY = 0.000000000001

def LofN(n):
    return range(n)

def LofNmax(nmax):
    return np.concatenate(tuple(LofN(n) for n in range(nmax + 1)))

#number of electrons in ith l, n combination
def LofI(i, nmax = 10):
    return int(LofNmax(nmax)[i])

#number of electrons vs. angular momentum quantum number
def degeneracy(l):
    return 2 * (2 * l  + 1)

#num electrons filled up to the ith n, l combination, only works up to 3p
def numElectrons(i, nmax = 10):
    perOrbital = map(degeneracy, LofNmax(nmax))
    return int(np.add.accumulate(perOrbital)[max(0, i)])

def sph_fft(x, y, qSamples):
    """
    Perform spherical Fourier transform of [x, y] and evaluate it at each 
    q in qSamples. 
    """
    def integrand(x, y): 
        return lambda q: y * (np.sin(q * x)/(q * x + TINY))
    def transform(q):
        return np.trapz(np.array([integrand(xx, yy)(q) for xx, yy in \
                zip(x, y)]), x = x)
    #ipdb.set_trace()
    return qSamples, np.array(map(transform, qSamples))


class Orbitals:
    def __init__(self, radii, real, imag, atomicN):
        """
        argument radii is expected to be in units of Bohr.
        """
        self.radii = BOHRTOA * radii
        self.real = real
        self.imag = imag
        self.atomicN = atomicN

    #index orbitals starting from 0
    def rho(self, norbit, degen = False):
        """
        incorporate degeneracy if degeneracy == True
        """
        #ipdb.set_trace()
        rl = self.real[norbit]
        im = self.imag[norbit]
        if degen == True:
            degen = max(min(degeneracy(LofI(norbit)), self.atomicN - \
                    numElectrons(norbit - 1)), 0)
        else:
            degen = 1
        return degen * (rl * rl + im * im)/BOHRTOA

    def shellFormFactor(self, qsamples, norbit, degeneracy = False):
        #TODO: there's some resundancy here, since we're already ensuring the correct
        #number of electrons in rho
        """
        evaluate form factor for a single orbital populated by 2 electrons.
        """
        if norbit > len(self.real) - 1:
            raise ValueError("orbital label larger than number of non-empty orbitals")
        density = self.rho(norbit, degen = degeneracy)
        return sph_fft(self.radii, density, qsamples)

    def formFactor(self, orbitMax, qsamples, degeneracy = False):
        """
        calculate total form factor.
        """
        facs = [self.shellFormFactor(norbit, qsamples, self.atomicN, degeneracy = degeneracy)[1] \
                for norbit in range(orbitMax)]
        return qsamples, np.sum(facs, axis = 0)
        
