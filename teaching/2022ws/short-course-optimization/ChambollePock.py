# This code implements a rudimentary version of the Chambolle-Pock algorithm.
def ChambollePock(x0, p0, prox_f, prox_gstar, A, tau, sigma, maxiter):
    # Set the initial guesses.
    x = x0
    p = p0
    pbar = p

    # Enter the main loop.
    for iter in range(maxiter):
        # Evaluate the proximity map of f.
        x = prox_f(tau, x - tau * A.T @ pbar)

        # Evaluate the proximity map of gstar.
        pold = p
        p = prox_gstar(sigma, p + sigma * A @ x)

        # Extrapolate the dual variable.
        pbar = 2 * p - pold

    # Return the result. 
    return x, p
