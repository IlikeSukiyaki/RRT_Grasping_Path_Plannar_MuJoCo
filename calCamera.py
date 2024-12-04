# import numpy as np
# from dm_control import mjcf
# from dm_control import viewer

# # 加载模型
# model = mjcf.from_file('/home/da/project2_deliver(1)/project2_deliver/assets/banana.xml')
# sim = model.get_sim()

# # 选择相机ID（假设你有一个相机名为 "camera"）
# camera_id = sim.model.camera_name2id("camera")

# # 获取相机的视角参数，包括位置和旋转
# camera_position = sim.data.cam_xpos[camera_id]
# camera_orientation = sim.data.cam_xmat[camera_id].reshape(3, 3)

# # 相机外参 (旋转矩阵和平移向量)
# rotation_matrix = camera_orientation
# translation_vector = camera_position

# # 打印结果
# print("相机旋转矩阵:\n", rotation_matrix)
# print("相机平移向量:\n", translation_vector)
from scipy.spatial.transform import Rotation as R
import numpy as np

# 使用欧拉角表示 pitch 旋转 90 度（以弧度为单位）
euler_angles = [np.pi/2, 0, 0]  # yaw, pitch, roll

# 转换为四元数
quat = R.from_euler('xyz', euler_angles).as_quat()  # 返回 [x, y, z, w]

# 打印结果，注意要把顺序变成 [w, x, y, z] 用于 XML
quat_xml_format = [quat[3], quat[0], quat[1], quat[2]]
print("四元数用于 XML:", quat_xml_format)
