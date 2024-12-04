from scipy.spatial.transform import Rotation as R
import numpy as np

# 定义四元数，顺序为 [w, x, y, z]
quat = [0, 0, 0, 1]  # 例如，一个单位四元数代表没有旋转

# 使用 Rotation 类将四元数转换为旋转矩阵
rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # 将 [w, x, y, z] 转换为 [x, y, z, w] 顺序
rotation_matrix = rotation.as_matrix()

# 打印旋转矩阵
print("旋转矩阵:")
print(rotation_matrix)