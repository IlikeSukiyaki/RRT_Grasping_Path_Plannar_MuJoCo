import numpy as np
from scipy.linalg import rq

# Given 3x4 camera matrix P
P = np.array([[127.6, -84.02490459, 95.7, -46.14003817],
              [45.18505725, 0, 138.91992367, -59.75],
              [0.8, 0, 0.6, -0.5]])

# Extract the 3x3 submatrix M
M = P[:, :3]

# Perform RQ decomposition to get intrinsic (K) and rotation (R)
K, R = rq(M)

# Normalize K to ensure positive diagonal
T = np.diag(np.sign(np.diag(K)))
K = K @ T
R = T @ R

# Extract the translation vector t
t = np.linalg.inv(K) @ P[:, 3]

print("Intrinsic matrix K:")
print(K)
print("\nRotation matrix R:")
print(R)
print("\nTranslation vector t:")
print(t)
