import numpy as np
import matplotlib.pyplot as mpl
import sys

points = np.loadtxt(sys.argv[1])
hull = np.loadtxt(sys.argv[2])
hull = list(hull)
hull.append(hull[0])
hull = np.array(hull)

mpl.plot(points[:,0], points[:,1], 'o')
mpl.plot(hull[:,0], hull[:,1], 'r-')

mpl.savefig("convex-hull.pdf")
