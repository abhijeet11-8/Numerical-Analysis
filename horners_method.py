import numpy as np

def horner(coef, x):
    out = 0
    for c in reversed(coef):
        # func = 2 + 3x + 4x^2
        out = out * x + c
    return out

c = np.array([2, 3, 1])

x_val = 1.0427

p_val = horner(c, x_val)
print("output = ", p_val)