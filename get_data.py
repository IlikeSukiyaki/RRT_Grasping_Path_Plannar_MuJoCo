import PIL.ImageShow
from dm_control import mujoco
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import PIL.Image
import itertools
import math
from scipy.spatial.transform import Rotation as R

# MuJoCo 模型路径
model_path = 'project2_deliver/assets/banana_bgr.xml'
physics = mujoco.Physics.from_xml_path(model_path)


def render_scene(sim):
    # 渲染 RGB 图像
    rgb_array = sim.render(camera_id=0, width=640, height=480)
    return rgb_array

def get_depth(sim):
    # depth is a float array, in meters.
    depth = sim.render(camera_id=0, height=480, width=640,depth=True)
    # Shift nearest values to the origin.
    depth -= depth.min()
    # Scale by 2 mean distances of near rays.
    depth /= 2*depth[depth <= 1].mean()
    # Scale to [0, 255]
    pixels = 255*np.clip(depth, 0, 1)
    image=PIL.Image.fromarray(pixels.astype(np.uint16))
    return image

def get_ground_truth_pose(sim):
    # 获取杯子的位姿（位置和旋转）
    pos = physics.named.data.xpos["cube"]# 获取位置
    quat=physics.named.data.xquat["cube"] # 获取旋转
    return pos, quat

def save_data(image_id,rgb_image, depth_image,mask_image):
    # 保存数据
    cv2.imwrite('project2_deliver/trainingset3/rgb/'+str(image_id)+'-color.png', rgb_image)
    # 假设 depth_image 是一个 NumPy 数组
    depth_image.save('project2_deliver/trainingset3/depth/'+str(image_id)+'-depth.png')
    mask_image.save('project2_deliver/trainingset3/mask_visib/'+str(image_id)+'-mask.png')

def get_mat(sim,object_name):
    box_mat = sim.named.data.geom_xmat[str(object_name)].reshape(3, 3)
    return box_mat

def get_bbx(sim,object_name):
    box_size = sim.named.model.geom_size[str(object_name)]
    box_pos = physics.named.data.geom_xpos[str(object_name)]
    box_mat = physics.named.data.geom_xmat[str(object_name)].reshape(3, 3)

    # 为每个维度计算正负偏移
    offsets = np.array([[-box_size[0], box_size[0]], 
                       [-box_size[1], box_size[1]], 
                      [-box_size[2], box_size[2]]])
    
    xyz_local = np.stack(list(itertools.product(*offsets))).T  # 形状为 (3, 8)
    xyz_global = box_pos[:, None] + box_mat @ xyz_local  # 转换为世界坐标系

    # Camera matrices multiply homogenous [x, y, z, 1] vectors.
    corners_homogeneous = np.ones((4, xyz_global.shape[1]), dtype=float)
    corners_homogeneous[:3, :] = xyz_global

    # Get the camera matrix.
    camera = mujoco.Camera(physics,camera_id=0)
    camera_matrix = camera.matrix
    print("camera_matrix",camera_matrix)
    xs, ys, s = camera_matrix @ corners_homogeneous
    # x and y are in the pixel coordinate system.
    x = xs / s
    y = ys / s
    #2d_pose
    twoD_pose=np.array((x, y)).T
    x_min, y_min = twoD_pose.min(axis=0)
    x_max, y_max = twoD_pose.max(axis=0)

    # pixels = camera.render()
    # fig, ax = plt.subplots(1, 1)
    # ax.imshow(pixels)
    # ax.plot(x, y, '+', c='w')
    # ax.set_axis_off()
    # plt.show()
    return x_min,y_min,x_max,y_max

def get_mask(sim):

    # 渲染分割图像
    seg = physics.render(camera_id=0,segmentation=False)

    # 提取对象 ID
    geom_ids = seg[:, :, 0]
    # Infinity is mapped to -1
    geom_ids = geom_ids.astype(np.float64) + 1

    # 假设我们要提取 ID 为 target_geom_id 的对象
    target_geom_id =14   # 替换为你感兴趣的对象 ID

    # 创建掩码，只保留目标对象
    mask = (geom_ids == target_geom_id).astype(np.float64)

    # 将掩码缩放到 [0, 1] 并映射到 [0, 255]
    pixels = 255 * mask
    pixels = pixels.astype(np.uint8)

    # 使用 OpenCV 调整掩码图像的大小为 640x480
    mask_resized = cv2.resize(pixels, (640, 480), interpolation=cv2.INTER_NEAREST)

    # 创建和显示特定对象的分割图像
    segmentation_image = PIL.Image.fromarray(mask_resized)
    return segmentation_image

def update_mug_position_and_quat(position, quaternion):
    physics.data.qpos[8:11]= position
    physics.data.qpos[11:15]= quaternion

def euler_to_quaternion(roll, pitch, yaw):
    """将欧拉角转换为四元数。

    参数:
    roll -- 绕X轴旋转的角度（弧度）
    pitch -- 绕Y轴旋转的角度（弧度）
    yaw -- 绕Z轴旋转的角度（弧度）

    返回:
    四元数（q_x, q_y, q_z, q_w）
    """
    # 计算半角的正弦和余弦
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    # 计算四元数
    q_w = cr * cp * cy + sr * sp * sy
    q_x = sr * cp * cy - cr * sp * sy
    q_y = cr * sp * cy + sr * cp * sy
    q_z = cr * cp * sy - sr * sp * cy

    return np.array([q_x, q_y, q_z, q_w])


def get_Rs(model_rotation):
        
   

    camera_rotation = physics.named.data.cam_xmat['right_pillar'].reshape(3, 3)  # 获取相机的旋转矩阵
    print(camera_rotation)
    # 创建模型到相机的变换矩阵
    # 旋转部分是相机旋转的逆
    rotation_matrix = np.linalg.inv(camera_rotation)
    R_object_to_camera = np.dot(rotation_matrix, model_rotation)  # 物体到相机的旋转矩阵
    return R_object_to_camera

def get_Ts(model_rotation):

    model_position = physics.named.data.xpos["banana"]  # 获取模型的位置信息



    # 获取相机的位置和旋转（假设相机名为 'my_camera'）
    camera_name = 'right_pillar'
    camera_position = physics.named.data.cam_xpos[camera_name]  # 获取相机的位置信息
    camera_rotation = physics.named.data.cam_xmat[camera_name].reshape(3, 3)  # 获取相机的旋转矩阵

    # 创建模型到相机的变换矩阵
    # 旋转部分是相机旋转的逆
    rotation_matrix = np.linalg.inv(camera_rotation)

    # 平移部分是模型位置减去相机位置经过相机旋转的变换
    translation_vector = model_position - (rotation_matrix @ camera_position)

    return translation_vector


def rotation_matrix_to_euler(R):
    """
    从旋转矩阵 R (3x3) 中提取欧拉角 (roll, pitch, yaw)
    
    参数:
    R -- 3x3 旋转矩阵
    
    返回:
    roll, pitch, yaw -- 欧拉角（以弧度为单位）
    """
    # 防止计算中的数值误差导致的 arcsin 函数的域外错误
    pitch = np.arcsin(-R[2, 0])
    
    # 如果 pitch = ±90°，则发生万向节锁（gimbal lock），此时 roll 和 yaw 无法唯一确定
    if np.cos(pitch) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = 0
        yaw = np.arctan2(-R[1, 2], R[1, 1])
    
    return roll, pitch, yaw

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    将欧拉角（roll, pitch, yaw）转换为旋转矩阵。
    
    参数:
    roll -- 绕 X 轴旋转的角度（弧度）
    pitch -- 绕 Y 轴旋转的角度（弧度）
    yaw -- 绕 Z 轴旋转的角度（弧度）
    
    返回:
    R -- 旋转矩阵（3x3）
    """
    # 绕 X 轴的旋转矩阵
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # 绕 Y 轴的旋转矩阵
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # 绕 Z 轴的旋转矩阵
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # 综合旋转矩阵：R = R_z * R_y * R_x
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

roll=0.0
rotation_increment=np.radians(5)

# print(physics.named.data.xmat["red_box"])
# print(physics.named.data.xquat["red_box"])
# print(euler_to_quaternion(0,0,30))
for id in range(10):
    camera = mujoco.Camera(physics,camera_id=0)
    matrix = camera.matrices()
    camera_matrix = camera.matrix
    #randomly make position
    if np.random.rand() > 0.5:
        random_posx=np.random.uniform(0.05,0.25)
    else:
        random_posx=np.random.uniform(-0.05,-0.2)
    
    if np.random.rand()>0.5:
        random_posy=np.random.uniform(0.25,0.4)
    else:
        random_posy=np.random.uniform(0.55,0.8)
    
    random_posz=0.05
    # model_position = physics.named.data.xpos["banana"]  # 获取模型的位置信息

    # position=model_position
    position=[random_posx,random_posy,random_posz]

    roll = np.random.uniform(0, 2 * np.pi)
    quaternion=euler_to_quaternion(0,0,np.pi/2)
    model_rotation=euler_to_rotation_matrix(0,0,np.pi/2)
    # print("yaw:",yaw)


    update_mug_position_and_quat(position, quaternion)
    physics.forward()


    rgb_image = physics.render(camera_id=0, width=640, height=480)

    # rotation_matrix=get_Rs(model_rotation)
    # transition_matrix=get_Ts(model_rotation)

    #to the world aix
    rotation_matrix=model_rotation
    transition_matrix=position

    transformation_matrix = np.eye(4)  
    transformation_matrix[:3, :3] = rotation_matrix  
    transformation_matrix[:3, 3] = transition_matrix
    # print(id)

    depth_image = get_depth(physics)  

    mask_image=get_mask(physics)

    object_id= str(id).zfill(6)
    save_data(object_id,rgb_image, depth_image,mask_image)  # 保存数据
     # Get the camera matrix.
    
    with open(f'project2_deliver/trainingset3/txt/{object_id}.txt', 'w') as myfile:
        np.savetxt(myfile, transformation_matrix)
    