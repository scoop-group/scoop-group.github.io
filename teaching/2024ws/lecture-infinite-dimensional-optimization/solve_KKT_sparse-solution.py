"""
Example solution file for Exercise 14.1 of the "Infinite Dimensional Optimization" class of WS 2024 at the University of Heidelberg
Lecture: Roland Herzog
Exercises: Georg Müller
"""

# Import fenics and plot functionality
from __future__ import print_function
from fenics import *
import numpy as np
import scipy as sc
import scipy.sparse as sps
import matplotlib.pyplot as plt
import math, time
from mpl_toolkits.mplot3d import Axes3D  # necessary for 3D plotting

# Define empty problem class
class problem:
    pass

def assemble_problem_data(prob):
	''' Assemble and collect problem data in a structure'''

	# Create mesh and define function space
	nx = math.ceil((prob.topRight[0]-prob.lowerLeft[0])/prob.h)
	ny = math.ceil((prob.topRight[1]-prob.lowerLeft[1])/prob.h)
	mesh = RectangleMesh(Point(prob.lowerLeft[0],prob.lowerLeft[1],0), 
	                     Point(prob.topRight[0],prob.topRight[1],0), 
	                     nx, ny,'right')
	V = FunctionSpace(mesh, 'P', 1)

	# Save mesh data in the problem structure
	prob.mesh = mesh
	prob.V    = V
	prob.v2d  = vertex_to_dof_map(prob.V) # Reordering map for vertex to dof indices
	prob.nx   = nx
	prob.ny   = ny

	# Assemble (bi)linear forms
	y = TrialFunction(V)
	v = TestFunction(V)
	prob.S = assemble(dot(grad(y),grad(v))*dx + dot(y,v)*ds)
	prob.M = assemble(dot(y,v)*dx)

	# Obtain backend types of fenics objects to convert to sparse matrices later
	prob.S = as_backend_type(prob.S).mat()  
	prob.M = as_backend_type(prob.M).mat()

	# Convert to the sparse csr_format for accelerated solving
	prob.S = sps.csr_matrix(prob.S.getValuesCSR()[::-1],shape=prob.S.size)
	prob.M = sps.csr_matrix(prob.M.getValuesCSR()[::-1],shape=prob.M.size)

	return prob

def solve_floor_heating_PDE(prob,f):
	''' Solve the floor heating PDE constraint for given right hand side f'''
	# Define rhs of variational problem and assemble
	v = TestFunction(prob.V)
	L = f*v*dx
	b = assemble(L)

	# Get sparse backend format
	b = as_backend_type(b)

	# Compute solution via sparse linear system solve
	y = sps.linalg.spsolve(prob.S, b)
	return y


def solve_floor_heating(prob, gamma):
	''' Solve the floor heating optimal control problem by solving the KKT conditions use a sparse direct linear system solver '''
	try:
		# Compute number of dofs
		nDoF = prob.M.shape[0]

		# Set right hand side of the optimality system
		zeros = sps.csr_matrix((1,nDoF))
		RHS = [prob.M.dot(prob.yd),
		       zeros,
		       zeros]

		# Set system matrix of the optimality system
		zeros = sps.csr_matrix((nDoF,nDoF))
		MAT = [[prob.M, zeros        ,  sps.csr_matrix.transpose(prob.S)],
		       [zeros , gamma*prob.M , -sps.csr_matrix.transpose(prob.M)],
		       [prob.S, -prob.M      ,  zeros]]
		
		# Sparsify problem matrices for faster solving
		MAT = sps.bmat(MAT,"csr")
		RHS = sps.csr_matrix.transpose(sps.hstack(RHS,"csr"))
		
		# Solve the ystem
		x = sps.linalg.spsolve(MAT,RHS)

		# Extract solution parts. Third part is the adjoint state.
		prob.ysols.append(x[prob.v2d])
		prob.usols.append(x[prob.v2d+nDoF])
	except:
		print('>>>>> Your floor heating solution is not working correctly.')
	
def plotSols(prob):
	''' Plot the results of a solved problem '''

	# Obtain data for desired state    
	ydVals = prob.yd[prob.v2d]

	# Obtain mesh data and reshape for plotting
	coords  = prob.mesh.coordinates()
	X = np.reshape(coords[:,0], [prob.ny+1,prob.nx+1])
	Y = np.reshape(coords[:,1], [prob.ny+1,prob.nx+1])

	fig = plt.figure()
	fig.suptitle('Optimal states and desired state')
	ax = []

	# Plot solutions for all gammas
	for i, gamma in enumerate(prob.gammas):
		ax.append(fig.add_subplot(2, math.ceil((len(prob.gammas)+1)/2), i+1, projection='3d'))
		try:
			Z = np.reshape(prob.ysols[i], [prob.ny+1, prob.nx+1])
			ax[-1].plot_surface(X,Y,Z)
			ax[-1].set_title('Gamma = %2.2e' % gamma)
		except:
			print('Missing data, I\'m skipping a plot.')
		zmin = 0.9*min([val for sol in prob.ysols + [ydVals] for val in sol ])
		zmax = 1.1*max([val for sol in prob.ysols + [ydVals] for val in sol ])
		ax[-1].set_zlim(zmin,zmax)

	# Plot desired state
	Z = np.reshape(ydVals,[prob.ny+1,prob.nx+1])
	ax.append(fig.add_subplot(2, math.ceil((len(prob.gammas)+1)/2), len(prob.gammas)+1, projection='3d'))
	ax[-1].plot_surface(X,Y,Z)
	ax[-1].set_title('Desired state')
	ax[-1].set_zlim(zmin,zmax)

#---------------- Start of main ------------------
# Define general data
probA = problem()
probA.lowerLeft = (0,0) # Unit square as
probA.topRight  = (1,1) # the domain
probA.h = 0.02          # Grid fineness

# Assemble problem data in advance
startAss = time.time()
probA = assemble_problem_data(probA)
endAss = time.time()
print('Assembly time: %f seconds' % (endAss-startAss))

# Set the regularization parameters
probA.gammas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

# Solve two different problem settings for the same regularization parameters and plot results
for ud in [Constant(1.0), Expression('sin(2*pi*x[0])*cos(2*pi*x[1])', degree=1)]:
	
	# Compute desired state from desired control
	startYdSolve = time.time()
	probA.yd = solve_floor_heating_PDE(probA, ud); 
	endYdSolve = time.time()
	print('Solve time for PDE constraint: %f seconds' % (endYdSolve-startYdSolve))
	
	probA.ud = ud
	probA.ysols = []
	probA.usols = []
	
	# Solve the optimal control problem
	start_Opt_Solves = time.time()
	for i in range(len(probA.gammas)):
	    solve_floor_heating(probA, probA.gammas[i]) # Solve optimization problem
	endOptSolves = time.time()
	print('Average optimization time: %f seconds' % ((endOptSolves-start_Opt_Solves)/len(probA.gammas)))
	
	# Plot
	plotSols(probA)

# Actually show all plots at the same time
plt.show()
