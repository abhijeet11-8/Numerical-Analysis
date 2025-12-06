import numpy as np

def newton_divided_diff(x, y):
    '''
    Compute coefficients of newton's interpolating polynomials.
    x, y: arrays of data points
    returns: array of coefficients [c0, c1, ..., cn-1]
    '''

    n = len(x)
    coef = np.copy(y).astype(float)

    # Compute divided differences
    for j in range(1, n):
        coef[j: n] = (coef[j:n] - coef[j-1:n-1])/(x[j:n] - x[0:n-j])

    return coef

def newton_polynomial(x_data, coef, x):
    '''
    Evaluate newton interpolating polynomial at x.
    '''

    n = len(coef)
    p = coef[-1]
    for k in range(1, n):
        p = coef[-1 - k] + (x - x_data[-1 -k]) * p
    return p


if __name__ == "__main__":
    A = np.array([
        [-2, 3],
        [0, 1],
        [2, 5]
    ], dtype=float)

    x = A[:, 0]
    y = A[:, 1]

    coef = newton_divided_diff(x, y)
    print("Coeficients ", coef)

    p = newton_polynomial(x, coef, x=0.5)
    print("value at x = 0.5 is  ", p)

    s = np.sum(coef)
    print("sum = ", s)