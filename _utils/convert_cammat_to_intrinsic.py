import numpy as np
from scipy.linalg import rq

# Camera matrix P
P = np.array([[127.6, -84.02490459, 95.7, -46.14003817],
              [45.18505725, 0., 138.91992367, -59.75],
              [0.8, 0., 0.6, -0.5]])
# P = np.array([[  84.02490459,    0.        ,  159.5       , -105.81245229],
#        [   0.        ,  -84.02490459,  119.5       ,  -14.19003817],
#        [   0.        ,    0.        ,    1.        ,   -0.4       ]])

# Extract M (3x3 submatrix)
M = P[:, :3]

# Perform RQ decomposition
K, R = rq(M)

# Ensure K has positive diagonal elements
T = np.diag(np.sign(np.diag(K)))
K = K @ T
R = T @ R

# Compute translation vector t
t = np.linalg.inv(K) @ P[:, 3]

print("Intrinsic Matrix K:\n", K)
print("Rotation Matrix R:\n", R)
print("Translation Vector t:\n", t)