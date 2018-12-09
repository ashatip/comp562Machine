#https://stackoverflow.com/questions/32046582/spline-with-constraints-at-border

import numpy as np
from scipy.interpolate import UnivariateSpline, splev, splrep
from scipy.optimize import minimize

def guess(x, y, k, s, w=None):
    """Do an ordinary spline fit to provide knots"""
    return splrep(x, y, w, k=k, s=s)

def err(c, x, y, t, k, w=None):
    """The error function to minimize"""
    diff = y - splev(x, (t, c, k))
    if w is None:
        diff = np.einsum('...i,...i', diff, diff)
    else:
        diff = np.dot(diff*diff, w)
    return np.abs(diff)

def spline_neumann(x, y, k=3, s=0, w=None):
    t, c0, k = guess(x, y, k, s, w=w)
    x0 = x[0] # point at which zero slope is required
    con = {'type': 'eq',
           'fun': lambda c: splev(x0, (t, c, k), der=1),
           #'jac': lambda c: splev(x0, (t, c, k), der=2) # doesn't help, dunno why
           }
    opt = minimize(err, c0, (x, y, t, k, w), constraints=con)
    copt = opt.x
    return UnivariateSpline._from_tck((t, copt, k))




import matplotlib.pyplot as plt
n = 20
x = np.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19])
y = np.array([0.0, 0.0, 0.0, 0.0, -5.12222e-05, -0.00021168, -0.000472569, -0.000797554, -0.00118193, -0.00153628, -0.00178105, -0.00187273, -0.00176204, -0.00133082, -0.000552282, 0.000569896, 0.00203041, 0.00382237, 0.00593729, 0.0083652])
std = 0.00000001
k = 3

sp0 = UnivariateSpline(x, y, k=k, s=n*std)
sp = spline_neumann(x, y, k, s=n*std)

plt.figure()
X = np.linspace(x.min(), x.max(), len(x)*10)

#plt.plot(X, sp0(X), '-r', lw=1, label='guess')
plt.plot(X, sp(X), '-r', lw=2, label='spline')

plt.plot(x, y, 'ok', label='data')

plt.legend(loc='best')
plt.show()