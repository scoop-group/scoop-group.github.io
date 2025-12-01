# This function creates a sparse matrix of horizontal and vertical differences between
# neighboring pixels for an image of size n1 (vertical) times n2 (horizontal).
import scipy.sparse 
def create_difference_matrix(n1, n2):
    # Initialize the lists holding the non-zero entries (rows, columns, values).
    rows = []
    cols = []
    vals = []

    # Create the entries pertaining to the tails of horizontal differences.
    how_many = n1 * (n2-1)
    rows = rows + [single_index(i1, i2+0, n1) for i1 in range(n1) for i2 in range(n2-1)]
    cols = cols + list(range(how_many))
    vals = vals + [-1] * how_many

    # Create the entries pertaining to the heads of horizontal differences.
    rows = rows + [single_index(i1, i2+1, n1) for i1 in range(n1) for i2 in range(n2-1)]
    cols = cols + list(range(how_many))
    vals = vals + [+1] * how_many

    # Create the entries pertaining to the tails of vertical differences.
    how_many = n2 * (n1-1)
    ncols = max(cols)
    rows = rows + [single_index(i1+0, i2, n1) for i2 in range(n2) for i1 in range(n1-1)]
    cols = cols + list(range(ncols + 1, ncols + 1 + how_many))
    vals = vals + [-1] * how_many

    # Create the entries pertaining to the heads of vertical differences.
    rows = rows + [single_index(i1+1, i2, n1) for i2 in range(n2) for i1 in range(n1-1)]
    cols = cols + list(range(ncols + 1, ncols + 1 + how_many))
    vals = vals + [+1] * how_many

    # Create the sparse matrix A transpose, but return A.
    AT = scipy.sparse.coo_matrix((vals, (rows, cols)))
    return AT.T

# This function returns a single index from a pair of indices (i1, i2).
# The convention is that i1 is the least significant index, i.e., matrices are
# stacked column by column.
def single_index(i1, i2, n1):
    return i1 + n1 * i2
