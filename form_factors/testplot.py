# coding: utf-8
#get_ipython().magic(u'run RCNextract.py')
#get_ipython().magic(u'run -i extractWF.py')
import sys

mt = Orbitals(radii, real, imag, str(sys.argv[1]))
ffacs = [mt.shellFormFactor(x, n) for n in range(7)]
import matplotlib.pyplot as plt
#[plt.plot(*fac) for fac in ffacs]
yffacs = [ffac[1] for ffac in ffacs]
all = np.sum(yffacs, axis = 0)
plt.plot(x, all)
Fe = getFofQ('Zn')
plt.plot(x, Fe(x))
plt.show()
