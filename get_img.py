import numpy as np
import cv2
from dm_control import mujoco
from dm_control.mujoco import sim

# 加载模型
model_path = '/home/da/project2_deliver(1)/project2_deliver/assets/test.xml'  # 替换为你的 XML 文件路径
model = mujoco.MjModel.from_xml_path(model_path)
sim = sim.MjSim(model)

# 创建渲染器（用于获取图像）
renderer = mujoco.MjRenderContextOffscreen(sim, width=640, height=480)

# 渲染并获取 RGB 和深度图像
sim.step()  # 执行一步模拟（如果需要）

# 获取 RGB 图像
rgb_image = renderer.render(mode='rgb')
# 获取深度图像
depth_image = renderer.render(mode='depth')

# 转换深度图为 0-255 范围（深度图是浮动值）
depth_image_normalized = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image)) * 255
depth_image_normalized = depth_image_normalized.astype(np.uint8)

# 保存 RGB 图像
cv2.imwrite('rgb_image.png', rgb_image)

# 保存深度图像
cv2.imwrite('depth_image.png', depth_image_normalized)

print("RGB 和深度图像已保存。")
