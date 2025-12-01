# This code solves a minimal cost network flow problem.

# Resolve the dependencies.
from scipy.optimize import linprog
import numpy as np
import networkx as nx

# Construct the digraph using vertices and edges. Notice that networkx may
# shuffle the edges, so we attach the cost to the edges.
vertices = range(1,10)
edges = [(1,4), (1,5), (2,4), (2,5), (3,4), (3,5), (4,5), (4,6), (4,7), (5,8), (5,9)]
costs = np.array([0.8, 2.0, 2.5, 1.0, 1.2, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
edgesWithCosts = [(v1, v2, {"costs": f'{c}'}) for ((v1, v2), c) in zip(edges, costs)] 
G = nx.DiGraph()
G.add_nodes_from(vertices)
G.add_edges_from(edgesWithCosts)

# Setup the incidence matrix. 
A = nx.incidence_matrix(G, oriented = True)
A = A.toarray()

# Retrieve the cost vector from the edge data in the order stored in the digraph.
c = list(zip(*G.edges.data('costs')))[2]

# Setup the vector of vertex balances.
b = np.array([-100, -200, -300, 0, 0, 150, 150, 150, 150])

# Setup the lower and upper bounds.
bounds = [(0, None) for j in edges]

# Call linprog to solve the problem.
result = linprog(c, A_eq = A, b_eq = b, bounds = bounds, method = 'simplex')

# Check the success flag.
import sys
if not result.success:
  print(result.message)
  sys.exit()

# Attach attributes to the graph's edges.
edgeFlow = dict(zip(G.edges, result.x))
edgeFlowAtCosts = dict([((v1,v2), f'{x} @ {c}')  
  for (v1, v2, c), x in zip(G.edges.data('costs'), result.x)]) 

# Layout the digraph (assign vertex positions).
positions = nx.nx_agraph.graphviz_layout(G, prog = "neato")

# Resolve further dependencies.
import matplotlib.pyplot as plt
import tikzplotlib

# Show and export the digraph showing the optimal flow.
plt.figure(figsize = (10,10))
nx.draw(G, positions, with_labels = True, node_color = [[0.8] * 3], node_size = 1500)
nx.draw_networkx_edge_labels(G, positions, edge_labels = edgeFlowAtCosts)
#  plt.savefig("../graphs/solveOptimalTransport.pdf")
#  tikzplotlib.save("../graphs/solveOptimalTransport.tex")
plt.show()
