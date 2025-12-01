# This script visualizes the chaotic convergence behavior of undamped Newton-Raphson for finding zeros of complex function (rewritten as problems in R^2)
# Plot colors R^2 by root reached when the point in question is chosen as the initial value

import numpy as np
from local_newton_root import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cmath
import time
import sys
import pdb

plt.rcParams["figure.figsize"] = (10,10)

# More general version down below (this one is more efficient though)
# Define f:R^2 -> R^2 corresponding to the complex function z -> z^3-1
def f3(x, derivatives):
	values = {}

	if derivatives[0]:
		values["function"] = TODO
	if derivatives[1]:
		values["derivative"] = TODO
	return values

def get_rootIndex(roots, root, norm):
	# Compute list of norm distances
	dists = [norm(rt - root) for rt in roots]
	
	# Get the index of r in the list roots. If r is not in roots, append it to the list.
	try:
		return next(index for index, dist in enumerate(dists) if dist < 1e-5)
	except StopIteration:
		roots.append(root)
		return len(roots) - 1

def plot_newton_fractal(f, nd, newtonParameters):
	# Set list of colors to distinguish roots
	colors = ['b', 'g', 'r', 'c', 'm', 'y']

	# Set boundary of domain
	x1min, x1max, x2min, x2max = (-1.2, 1.2, -1.2, 1.2)

	# Make container for root indices that will be plotted
	rootIndices = np.zeros((nd, nd))

	# Make container for iterations needed
	iters = - np.ones((nd, nd))

	# Initialize empty container for roots found
	roots = []

	preconditioner = np.eye(2)

	# Status indicator
	oldPercent = 0

	# Iterate over discretized domain "columnwise" upwards
	for ix, x1 in enumerate(np.linspace(x1min, x1max, nd)):
		for iy, x2 in enumerate(np.linspace(x2min, x2max, nd)):

			# Print status
			percent = np.round(100 *(ix*nd + iy) / (nd**2))
			if percent >= oldPercent+5:
				print('Computations at %d percent' % percent)
				oldPercent = percent
				sys.stdout.flush()

			# Set the initial value
			x0 = np.array([x1,x2])

			# Apply newton to find a root TODO Your Newton might be called differently
			res = newton_root(f, x0, preconditioner, parameters=newtonParameters)

			# Check termination flag of newton
			# TODO make sure your local Newton has the correct termination flags
			match res['exitflag']:
				case 4:
					print('Warning: Newton failed to compute an update direction (no convergence).')
					# Set number of iterations to max iterations
					iterations = newtonParameters['max_iterations']
					# Set any root index, as it will be plotted transparent anyway
					rootIndex = 0
				case 3:
					print('Warning: Newton exceeded the maximum number of iterations (no convergence).')
					# Set number of iterations to max iterations
					iterations = newtonParameters['max_iterations']
					# Set any root index, as it will be plotted transparent anyway
					rootIndex = 0
				case _:
					# Get the root and the iterations needed
					root = res['solution']
					# Get index of root
					rootIndex = get_rootIndex(roots, root, norm = lambda x: np.sqrt(x.dot(preconditioner.dot(x))))
					# Save number of iterations needed
					iterations = res['iter']

			# Save number of iterations needed
			iters[iy,ix] = iterations
			# Save index for plotting
			rootIndices[iy, ix] = rootIndex

	# See how many roots we found
	nroots = len(roots)

	if nroots > len(colors):
		# Use a "continuous" colormap if there are too many roots.
		cmap = 'hsv'
		alphas = np.ones(nd,nd)
	else:
		# Use a list of colors for the colormap: one for each root.
		cmap = ListedColormap(colors[:nroots])
		referenceIterations = int(np.max(np.ma.masked_array(iters,mask=iters==newtonParameters['max_iterations'])))
		alphas = 1-np.maximum(0,np.minimum(1,np.log(np.minimum(iters,referenceIterations))/np.log(referenceIterations)))


	fig = plt.figure()

	# Color the image
	plt.imshow(rootIndices, cmap=cmap, origin='lower', alpha = alphas, extent = [x1min, x1max, x2min, x2max])

	# Add an axis
	axis = plt.axis()

	# Mark the roots
	plt.plot([root[0] for root in roots], [root[1] for root in roots],'ko')

	plt.title(r'Newton fractal for root finding of $f(z)=z^{:d}-1$'.format(len(roots)))

	plt.xlabel(r'$x_1 = \Re(z)$')
	plt.ylabel(r'$x_2 = \Im(z)$')

	return fig

########################## Main Code ##############################################

# Set parameters for local newton
newtonParameters = {
	"atol_x" : 1e-10,
	"rtol_x" : 1e-10,
	"atol_f" : 1e-10,
	"rtol_f" : 1e-10,
	"max_iterations" : 1e3,
}

# Set the number of discretization points per dimension
nd = 151

print('Starting fractal plot computation.')

st = time.time()
fig = plot_newton_fractal(f3, nd, newtonParameters)
et = time.time()

print(f'Computation took {et-st} seconds')
	
plt.show()
