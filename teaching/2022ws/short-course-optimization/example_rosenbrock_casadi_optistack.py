# This code solves an unconstrained optimization problem with the Rosenbrock
# function using CasADi's python bindings, and CasADi's Opti interface for
# modeling of (un)constrained optimization problems.

# Resolve the dependencies.
from casadi import *

# Create an instance of the Opti interface.
opti = Opti()

# Create a 1-by-1 optimization variable (termed 'x') and another 1-by-1
# optimization variable (termed 'y').
x = opti.variable()
y = opti.variable()

# Set the fixed problem parameters.
a = opti.parameter()
b = opti.parameter()
opti.set_value(a, 1)
opti.set_value(b, 100)

# Setup the objective.
f = (a - x)**2 + b * (y - x**2)**2
opti.minimize(f)

# Setup the solver.
opti.solver('ipopt')

# Call the solver.
result = opti.solve()

# Show the solution.
print()
print(result.value(x), result.value(y))
