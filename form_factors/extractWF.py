from scipy.interpolate import interp1d
from numpy import fft
import numpy as np
from mu import mu
import pdb
import RCNextract

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
    perOrbital = list(map(degeneracy, LofNmax(nmax)))
    return int(np.add.accumulate(perOrbital)[max(0, i)])


class AtomicWF:
    def __init__(self, data_directory, elt_name, weights = None):
        # Wave function is assumed to be spherically symmetric, with zero
        # imaginary component.
        self.atomicN = mu.ElementData(elt_name).get_atomicN(elt_name)
        radii, real = RCNextract.import_WF(data_directory, self.atomicN)
        self.radii, self.real = radii, real
        self.density = self.real * self.real
        if weights is not None:
            assert len(weights) == len(self.real)
        self.weights = weights

    def shell_density(self, norbit, degen = False):
        """
        incorporate degeneracy if degeneracy == True

        Returns a function of q-vector
        """
        from utils.utils import extrap1d
        if degen == True:
            degen = max(min(degeneracy(LofI(norbit)), self.atomicN - \
                    numElectrons(norbit - 1)), 0)
        else:
            degen = 1
        weight = self.get_weight(norbit)
        ff = self.density[norbit] * weight * degen
        scalar = interp1d(self.radii, ff, bounds_error=False, fill_value=0.)
        def density_interp(radii):
            return scalar(radii)
        return density_interp

    def charge_density(self):
        orbitMax = len(self.real)
        density_orbitals = [self.shell_density(norbit) for norbit in range(orbitMax)]
        def density_sum(radii):
            return np.sum([ff(radii) for ff in density_orbitals], axis = 0)
        return density_sum

    def get_weight(self, norbit):
        if self.weights is None:
            return 1.
        return self.weights[norbit]

class AtomicFF:
    def __init__(self, elt_name, weights = None):
        """
        argument radii is expected to be in units of Bohr.
        """
        self.q, self.ff = get_Cowan_WFs(elt_name)
        #self.radii = BOHRTOA * self.radii
        self.imag = np.zeros_like(self.ff)
        self.atomicN = mu.ElementData(elt_name).get_atomicN(elt_name)
        if weights is not None:
            assert len(weights) == len(self.ff)
        self.weights = weights

    #index orbitals starting from 0
    def shell_form_factor(self, norbit, degen = False, weight = None):
        """
        incorporate degeneracy if degeneracy == True

        Returns a function of q-vector
        """
        #ipdb.set_trace()
        #im = self.imag[norbit]
        if degen == True:
            degen = max(min(degeneracy(LofI(norbit)), self.atomicN - \
                    numElectrons(norbit - 1)), 0)
        else:
            degen = 1
        if weight is None:
            weight = self.get_weight(norbit)
        ff = self.ff[norbit] * weight * degen
        scalar = interp1d(self.q, ff, bounds_error=False, fill_value=0.)
        def ff_interp(qvec):
            return scalar(np.linalg.norm(qvec))
        return ff_interp

    def get_form_factor(self, weights = None):
        orbitMax = len(self.ff)
        if weights is None:
            ff_orbitals = [self.shell_form_factor(norbit) for norbit in range(orbitMax)]
        else:
            ff_orbitals = [self.shell_form_factor(norbit, weight = weight) for norbit, weight in enumerate(weights)]
        def ff_sum(qvec):
            return np.sum([ff(qvec) for ff in ff_orbitals])
        return ff_sum


    def get_weight(self, norbit):
        """
        Get default weights (set at initialization time.
        """
        if self.weights is None:
            return 1.
        return self.weights[norbit]
#
#def structure_factor(form_factors, coordinates):
#    


def get_structure_factor(atoms, coordinates_list, weights_list):
    """
    Returns a function of q-vector
    """
    def structure_factor(qvec):
        AFFs = [AtomicFF(atom, weights = weights).get_form_factor()
                for atom, weights in zip(atoms, weights_list)]
        phases = [np.exp(-1j * np.dot(qvec, coords)) for coords in coordinates_list]
        #pdb.set_trace()
        return np.sum([phase * ff(qvec) for phase, ff in zip(phases, AFFs)])
    return structure_factor

def get_charge_density(atoms, coordinates_list, weights_list = None):
    if weights_list is None:
        weights_list = [None] * len(atoms)
    atoms_map = {'Mg': 'RCN/Mg', 'O': 'RCN/O'}
    def one_atom(atom, origin, weights):
        def density(position_vectors):
            shifted = position_vectors - origin
            radii = np.linalg.norm(shifted, axis = 1)
            return AtomicWF(atoms_map[atom], atom, weights = weights).charge_density()(radii)
        return density
    def total_density(position_vectors):
        #pdb.set_trace()
        FFs = [one_atom(atom, origin, weights)(position_vectors) for atom, origin, weights
               in zip(atoms, coordinates_list, weights_list)]
        return np.sum(FFs, axis = 0)
    return total_density

def G(basis, hkl):
    return basis * hkl

def get_hkl_sampler(atoms, coordinates_list, weights_list, reciprocal_basis):
    """
    Returns a function that takes hkl (as np array of length 3) and returns a diffracted intensity.
    """
    if weights_list is None:
        weights_list = [None] * len(atoms)
    sf = get_structure_factor(atoms, coordinates_list, weights_list)
    def i_hkl(hkl):
        G = np.dot(reciprocal_basis, hkl)
        amp = sf(G)
        return np.linalg.norm(amp)**2
    return i_hkl

def bgen(a):
    alpha = 2 * np.pi / np.dot(a[0], np.cross(a[1], a[2]))
    return np.array([np.cross(a[1], a[2]) * alpha,
                     np.cross(a[2], a[0]) * alpha,
                     np.cross(a[0], a[1]) * alpha])

def get_Cowan_WFs(elt_name):
    tab = np.genfromtxt('H-Kr_atomic_form_factors/%s_atomic_form_factor_by_orbital.txt' % elt_name).T
    q, ff = tab[0], tab[1:-1]
    return q, ff

def run_mgo():
    a = 4.212
    A = np.array(
                [np.array([0.5, 0.5, 0.0]) * a,
                np.array([0.0, 0.5, 0.5]) * a,
                np.array([0.5, 0.0, 0.5]) * a]
    )
    b = bgen(A)
