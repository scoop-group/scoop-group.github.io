# This code solves a parameter estimation problem for a given model, based on a
# set of measurement pairs, using CasADi's python bindings for modeling of
# (un)constrained optimization problems.

# Resolve the dependencies.
from casadi import *
import matplotlib.pyplot as plt

# Create the unknown parameters as optimization variables.
xy = MX.sym('xy', 2, 1)

# Specify the pairs of measurements.
measurements = [
        ([ 8,  6],  38),
        ([-3, -3], 220),
        ([ 1,  0], 222),
        ([ 8, -3], 300),
        ]

# Declare the beacon position as a variable.
beacon = MX.sym('beacon position', 2, 1)

# Specify the model function.
angle = fmod(atan2(beacon[1] - xy[1], beacon[0] - xy[0]) + 2*pi, 2*pi) * 180 / pi
model = Function('airplane', [xy, beacon], [angle], ['xy', 'beacon'], ['angle']) 

# Assemble the least-squares objective function.
f = 0
for i in range(len(measurements)):
    residual = model(xy = xy, beacon = measurements[i][0])['angle'] - measurements[i][1]
    f = f + residual**2
f = 0.5 * f

# Setup the initial guess.
x0 = [0, 0]

# Setup the solver.
problem = {'x': xy, 'f': f}
solver = nlpsol('airplane', 'ipopt', problem)

# Call the solver.
result = solver(x0 = x0)

# Show the solution.
print()
print(result['x'])

# Create a plot.
plt.figure(1)
plt.xlabel("x")
plt.ylabel("y")
for (index, measurement) in enumerate(measurements):
    plt.plot(measurement[0][0], measurement[0][1], 'bo')
    plt.text(measurement[0][0] + 0.2, measurement[0][1] + 0.2, 'beacon {0:d}'.format(index))
plt.plot(float(result['x'][0]), float(result['x'][1]), 'ro')
#  plt.text(measurement[0][0], measurement[0][1], 'beacon {0:d}'.format(index))
plt.text(float(result['x'][0]) + 0.2, float(result['x'][1]) + 0.2, 'airplane')
plt.savefig('example_parameter_estimation.pdf')
