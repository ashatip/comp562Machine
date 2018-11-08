import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

x = np.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.,17.,18.,19])
y = np.array([0.0, 0.0, 0.0, 0.0, -5.12222e-05, -0.00021168, -0.000472569, -0.000797554, -0.00118193, -0.00153628, -0.00178105, -0.00187273, -0.00176204, -0.00133082, -0.000552282, 0.000569896, 0.00203041, 0.00382237, 0.00593729, 0.0083652])

t, c, k = interpolate.splrep(x, y, s=0.0000000008, k=3)
print('''\
t: {}
c: {}
k: {}
'''.format(t, c, k))
print('t: '+str(len(t)))
print('c: '+str(len(c)))
N = 100
xmin, xmax = x.min(), x.max()
xx = np.linspace(xmin, xmax, N)
spline = interpolate.BSpline(t, c, k, extrapolate=False)

# deploy = interpolate.PPoly.from_spline([t,c,k])
plt.plot(x, y, 'bo', label='Original points', markersize=2)
plt.plot(xx, spline(xx), 'r', label='BSpline')
plt.scatter(t,spline(t), c='g')
plt.grid()
plt.legend(loc='best')
plt.show()