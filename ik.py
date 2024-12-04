import numpy as np
import matplotlib.pyplot as plt
import ikpy.chain
import ikpy.utils.plot as plot_utils
from dm_control import mujoco
import time
import transformations as tf
import cv2

# Load the robot arm chain from the URDF file
my_chain = ikpy.chain.Chain.from_urdf_file("assets/a1_right.urdf")

# Load the MuJoCo model from XML
model = mujoco.Physics.from_xml_path('assets/a1_ik.xml')
# Number of runs
num_runs = 10

# Video settings
frame_width = 640
frame_height = 480
fps = 10
output_file = 'ik.mp4'

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Frames to hold each frame for 0.5 seconds
frames_per_step = int(0.5 * fps)

# Loop over the simulation runs
for run in range(num_runs):
    # Generate random target position
    target_position = np.random.rand(3) * 0.1 + np.array([-0.1, 0.5, 0.2])
    target_orientation_euler = np.random.rand(3) * np.pi * 0.2
    # target_orientation_euler = np.array([0, 0, 0])
    target_orientation = tf.euler_matrix(*target_orientation_euler)[:3, :3]
    print(f"Run {run+1}: {target_position }, {target_orientation }")

    # Compute the joint angles required to reach the target position
    joint_angles = my_chain.inverse_kinematics(target_position, target_orientation, "all")
    print("The angles of each joint are:", joint_angles)

    # Compute the real frame position based on the computed joint angles
    real_frame = my_chain.forward_kinematics(joint_angles)
    print("The real frame is:", real_frame)

    # Set the mocap position and orientation in the simulation
    model.data.mocap_pos[0] = target_position
    target_orientation_4x4 = np.eye(4)
    target_orientation_4x4[:3, :3] = target_orientation
    target_quaternion = tf.quaternion_from_matrix(target_orientation_4x4)
    model.data.mocap_quat[0] = target_quaternion

    # Set the joint angles of the robot arm in the simulation
    model.data.qpos[0:6] = joint_angles[1:]

    # Run a simulation step to update the state
    model.forward()

    # Render the frame from the simulation
    frame = model.render(camera_id=1, height=frame_height, width=frame_width)
    
    # Convert the frame to RGB format (if needed)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Write the same frame multiple times to hold for 0.5 seconds
    for _ in range(frames_per_step):
        video_writer.write(frame_rgb)

# Release the video writer
video_writer.release()

print(f"Video saved as {output_file}")
