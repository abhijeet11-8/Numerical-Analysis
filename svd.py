import numpy as np
import numpy.linalg as la

A = np.array([[6, 5, -5], [2, 6,  -2], [2, 5, -1]])
U, E, V = la.svd(A, compute_uv=True)
print(f"u = {U}, \n E = {E} \n V = {V}")
print(U @ np.diag(E) @ V)

n, m = A.shape
print(np.allclose(U.T @ U, np.identity(n)))
print(np.allclose(V.T @ V, np.identity(m)))

print(la.matrix_rank(A))
print(la.matrix_rank(A)==len(E))

# e (i)
## 2 norm of A
sorted = sorted(E)
print(sorted[0])

## forbenius norm of A
f_n = (np.sum(A**2))**0.5
print(f_n)

### condition no. of A k(A) = sigma_max/sigma_min
k_a = sorted[0]/sorted[-1]
print(k_a)

# Moore-peninrose pseudoinverse

pseudo_inv = la.pinv(A)
print(pseudo_inv)
print(np.allclose(A @ pseudo_inv @ A, A))


def rank_k_approximator(A:np.array, rank):
    U, E, V = la.svd(A, compute_uv=True)
    if rank <= 0:
        raise ValueError("Rank must be a positive integer.")
    
    # Ensure rank does not exceed the number of singular values
    k = min(rank, len(E))
    e = E[:k]
    u = U[:, :k]
    v = V[:k, :]

    return u @ np.diag(e) @ v

A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20]
])

# Perform rank-2 approximation
low_rank_A = rank_k_approximator(A, 1)

print("\n--- Rank-1 Approximation (A_k) (First 3 rows) ---")
print(low_rank_A[:3, :])

norm = la.norm(A-low_rank_A, 'fro')
print(norm)

