import numpy as np

def phi(x:np.array):
    return x[1]


def power_method(A:np.array, x:np.array, M:int):
    print(f"k = [0]-> x(0) = {x})")
    for k in range(1, M):
        y = A @ x
        r = phi(y)/phi(x)
        x = y / np.linalg.norm(y, np.inf)

        print(f"k = [{k}]-> x({k}) = {x}, r({k-1}) = {r}")

if __name__ == "__main__":
    A = np.array([[6, 5, -5], [2, 6,  -2], [2, 5, -1]])
    x = np.array([-1, 1, 1])
    power_method(A=A, x=x.T, M= 100)