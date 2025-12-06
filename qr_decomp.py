import numpy as np
import numpy.linalg as la

A = np.array([[6, 5, -5], [2, 6,  -2], [2, 5, -1]])
Q, R = la.qr(A)


# R @ x = Q.T @ b
b = np.array([-1, 1, 1])
y = Q.T @ b
x = la.solve(R, y)
print(x)

trace_q = la.trace(Q)
trace_r = la.trace(R)
print(f"trace Q = {trace_q}, trace R = {trace_r}")

r = b - A @ x
norm = la.norm(r, ord=2)
print(norm)