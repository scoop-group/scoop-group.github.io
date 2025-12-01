# This code solves an unconstrained optimization problem with the Rosenbrock
# function using CasADi's python bindings. For lack of a solver specializing
# in unconstrained optimization solver, the solver of choice is nlpsol with
# ipopt, see https://web.casadi.org/docs/#nonlinear-programming.

# Resolve the dependencies.
from casadi import *
import matplotlib.pyplot as plt
import numpy as np

# Create a 1-by-1 optimization variable (termed 'x') and another 1-by-1
# optimization variable (termed 'y').
# SX is the class in CasADi representing matrices composed of symbolic expressions.
# There is also MX for more complex expressions.
x = SX.sym('x', 1, 1)
y = SX.sym('y', 1, 1)

# Set the fixed problem parameters.
a = DM(1)
b = DM(100)

# Setup the objective.
f = (a - x)**2 + b * (y - x**2)**2

# Setup the solver.
problem = {'x': vertcat(x,y), 'f': f}
solver = nlpsol('production', 'ipopt', problem)

# Call the solver.
result = solver()

# Show the solution.
print()
print(result['x'])

# Evaluate the objective on a grid.
f_ = Function('f', [x, y], [f], ['x', 'y'], ['value'])
x_, y_ = np.meshgrid(np.linspace(a-1.5, a+0.5, 100), np.linspace(a**2-1.5, a**2+0.5, 100))
z_ = DM.zeros(x_.shape)
for i in range(x_.shape[0]):
    for j in range(x_.shape[1]):
        z_[i,j] = f_(x_[i,j], y_[i,j])

# Create a plot.
plt.figure(1)
plt.xlabel("x")
plt.ylabel("y")
plt.contourf(x_, y_, z_, levels = np.logspace(-2, 2, 50))
plt.plot(float(result['x'][0]), float(result['x'][1]), 'ro')
plt.set_cmap('cividis')
plt.savefig('example_rosenbrock.pdf')
