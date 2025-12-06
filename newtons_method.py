import numpy as np

def coefficients(x, y):
    '''
    Calculate the coefficients for the polynomials.
    '''

    n = len(x)
    y = np.copy(y).astype(float)

    for j in range(1, n):
        y[j:n] = (y[j:n] - y[j-1:n-1])/(x[j:n] - x[0, n-j])
    return y

def polynomial(x_data, coef, x):
    n = len(coef)
    p = coef[-1]

    for k in range(1, n):
        