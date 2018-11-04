########################################
############## PYTHON 3 ################
########################################

import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import CubicSpline
import numpy as np

# CHANGE THIS NUMBER TO REACH THAT SPECIFIC SAMPLE
samplenumber = 1;

with open('shapemap1000000.txt') as f:
	reader = csv.reader(f, delimiter=' ')
	# result=[[float(s) for s in row] for i,row in enumerate(reader) if i%3 == 2]
	#result=[row.strip() for i,row in enumerate(reader) if i < 5]
	for i in range(3*(samplenumber-1)): # skip the first n samples to reach the sample data we want
		next(reader)
	next(reader)
	next(reader)
	coords = [float(s) for s in next(reader)[:-1]] # skip whitespace at the end

print("done")
x = coords[::3]
y = coords[1::3]
z = coords[2::3]



fig2d = plt.figure()
ax2d = fig2d.add_subplot(111)

ax2d.scatter(range(20), x, c='r', label='x')
#ax2d.scatter(range(20), y, c='g', label='y')
#ax2d.scatter(range(20), z, c='b', label='z')
#ax2d.spines['top'].set_color('none')
#ax2d.spines['bottom'].set_position('center')
plt.axvline(x=0, c='black')
plt.axvline(x=19, c='black')


# fitx = np.polyfit(range(20), x, 4)
# plt.plot(range(20), np.polyval(fitx, range(20)))

spl = InterpolatedUnivariateSpline(range(20), x)
#spl = UnivariateSpline(range(20), x)
#spl.set_smoothing_factor(0)
plt.plot(range(20), spl(range(20)), 'g', lw=2)


plt.legend(loc='upper left');
plt.xlabel('t')
plt.show()


# PLOT x, Y, Z IN 3D
# fig3d = plt.figure()
# ax3d = fig3d.add_subplot(111, projection='3d')

# ax3d.scatter(x, y, z)

# ax3d.set_xlabel('X Label')
# ax3d.set_ylabel('Y Label')
# ax3d.set_zlabel('Z Label')

# plt.show()
