# Import necessary modules
from dm_control import mujoco
import cv2
import numpy as np
import random
import ikpy.chain
import transformations as tf
from PIL import Image
import sys

# Load dm_control model
model = mujoco.Physics.from_xml_path('assets/banana.xml')

# Load the robot arm chain from the URDF file
my_chain = ikpy.chain.Chain.from_urdf_file("assets/a1_right.urdf")


class RRT:
    class Node:
        def __init__(self, q):
            self.q = q
            self.path_q = []
            self.parent = None

    def __init__(self, start, goal, joint_limits, expand_dis=0.03, path_resolution=0.001, goal_sample_rate=50,
                 max_iter=1000):
        self.start = self.Node(start)
        self.end = self.Node(goal)
        self.joint_limits = joint_limits
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []

    def planning(self, model):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, model):
                self.node_list.append(new_node)

            if self.calc_dist_to_goal(self.node_list[-1].q) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(final_node, model):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None

    def get_nearest_node_index(self, node_list, rnd_node):
        """
        Find the index of the nearest node to the random node.

        Args:
            node_list: List of nodes in the RRT tree.
            rnd_node: Randomly generated node.

        Returns:
            Index of the nearest node in the node list.
        """
        dlist = [np.linalg.norm(np.array(node.q) - np.array(rnd_node.q)) for node in node_list]
        min_index = dlist.index(min(dlist))
        return min_index

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(np.array(from_node.q))
        distance = np.linalg.norm(np.array(to_node.q) - np.array(from_node.q))
        if extend_length > distance:
            extend_length = distance
        num_steps = int(extend_length / self.path_resolution)
        if num_steps == 0:
            num_steps = 1  # Ensure at least one step
        delta_q = (np.array(to_node.q) - np.array(from_node.q)) / num_steps

        for i in range(num_steps):
            new_q = new_node.q + delta_q
            new_node.q = np.clip(new_q, [lim[0] for lim in self.joint_limits], [lim[1] for lim in self.joint_limits])
            new_node.path_q.append(new_node.q.copy())

        new_node.parent = from_node
        return new_node

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rand_q = [random.uniform(joint_min, joint_max) for joint_min, joint_max in self.joint_limits]
        else:
            rand_q = self.end.q
        return self.Node(rand_q)

    def check_collision(self, node, model):
        return check_collision_with_dm_control(model, node.q)

    def generate_final_course(self, goal_ind):
        path = []
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.extend(reversed(node.path_q))
            node = node.parent
        path.append(self.start.q)
        return path[::-1]

    def calc_dist_to_goal(self, q):
        return np.linalg.norm(np.array(self.end.q) - np.array(q))


def get_end_effector_pose(joint_angles):
    """
    Calculates the pose of the end-effector given the joint angles.

    Args:
        joint_angles: List or array of joint angles.

    Returns:
        position: 3D position of the end-effector.
        orientation: 3x3 rotation matrix representing the orientation.
    """
    full_joint_angles = [0] + list(joint_angles)  # Include base joint if necessary
    frame_matrix = my_chain.forward_kinematics(full_joint_angles)
    position = frame_matrix[:3, 3]
    orientation = frame_matrix[:3, :3]
    return position, orientation


def check_collision_with_dm_control(model, joint_config):
    """
    Function to check if a given joint configuration results in a collision using dm_control's collision detection.
    Args:
        model: dm_control Mujoco model
        joint_config: List of joint angles to check for collision
    Returns:
        True if collision-free, False if there is a collision
    """
    model.data.qpos[0:6] = joint_config  # Set joint positions
    model.forward()  # Update the simulation state

    # Check for collisions
    contacts = model.data.ncon  # Number of contacts (collisions)
    # contacts=0
    return contacts == 0 or check_gripper_collision(model)  # True if no contacts (collision-free)


def check_gripper_collision(model):
    all_contact_pairs = []
    for i_contact in range(model.data.ncon):
        id_geom_1 = model.data.contact[i_contact].geom1
        id_geom_2 = model.data.contact[i_contact].geom2
        name_geom_1 = model.model.id2name(id_geom_1, 'geom')
        name_geom_2 = model.model.id2name(id_geom_2, 'geom')
        contact_pair = (name_geom_1, name_geom_2)
        all_contact_pairs.append(contact_pair)
    touch_banana_right = ("a1_right/a1_8_gripper_finger_touch_right", "banana_collision") in all_contact_pairs
    touch_banana_left = ("a1_right/a1_8_gripper_finger_touch_left", "banana_collision") in all_contact_pairs
    return touch_banana_left or touch_banana_right


def arrange_frames_in_grid(frames, frames_per_row, width, height):
    """
    Arrange a list of frames into a grid.

    Args:
        frames: List of frames (images) to arrange.
        frames_per_row: Number of frames per row in the grid.
        width: Width of each frame.
        height: Height of each frame.

    Returns:
        A single frame containing all frames arranged in a grid.
    """
    num_frames = len(frames)
    num_rows = int(np.ceil(num_frames / frames_per_row))
    grid_rows = []

    for row_idx in range(num_rows):
        start_idx = row_idx * frames_per_row
        end_idx = min(start_idx + frames_per_row, num_frames)
        row_frames = frames[start_idx:end_idx]
        # If this row has fewer frames, pad with black images
        if len(row_frames) < frames_per_row:
            padding_frames = [np.zeros((height, width, 3), dtype=np.uint8)] * (frames_per_row - len(row_frames))
            row_frames.extend(padding_frames)
        row = np.concatenate(row_frames, axis=1)
        grid_rows.append(row)

    frame_grid = np.concatenate(grid_rows, axis=0)
    return frame_grid


def apply_rrt_path_to_dm_control(model, path, video_name="rrt_robot_motion_with_transfer.mp4",
                                 pose_log_file="end_effector_poses_with_transfer.txt"):
    """
    Applies the RRT-generated path to the simulation, records frames into a video,
    and logs the end-effector poses into a text file.

    Args:
        model: dm_control Mujoco model.
        path: List of joint configurations generated by the RRT planner.
        video_name: Name of the output video file.
        pose_log_file: Name of the output text file for end-effector poses.
    """
    # Setup for video recording
    width, height = 640, 480  # Resolution of each camera
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4

    # Define camera IDs and names
    camera_names = ['front', 'right', 'back', 'left']  # Names of the cameras defined in banana.xml
    camera_ids = [model.model.name2id(name, 'camera') for name in camera_names]
    frames_per_row = 2  # Arrange frames in a 2x2 grid

    num_cameras = len(camera_ids)
    num_rows = int(np.ceil(num_cameras / frames_per_row))

    output_width = width * frames_per_row
    output_height = height * num_rows

    out = cv2.VideoWriter(video_name, fourcc, 20.0, (output_width, output_height))

    # Initialize list to store end-effector poses
    end_effector_poses = []

    # Set initial joint angles
    model.data.qpos[0:6] = start
    model.forward()

    # Apply the path to the simulation and record the video
    for q in path:
        model.data.ctrl[0:6] = q  # Control inputs for the joints

        # Render from all cameras
        frames = []
        for cam_id in camera_ids:
            frame = model.render(camera_id=cam_id, width=width, height=height)
            frames.append(frame)

        # Arrange frames in a 2x2 grid
        frame_grid = arrange_frames_in_grid(frames, frames_per_row, width, height)

        # Convert frame from RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR)

        # Write the frame to the video
        out.write(frame_bgr)

        # Record the end-effector pose
        position, orientation = get_end_effector_pose(q)
        orientation_flat = orientation.flatten()
        pose = np.concatenate((position, orientation_flat))
        end_effector_poses.append(pose)

        # Step the simulation forward
        model.step()

    # Pause for a moment to stabilize
    for i in range(50):
        # Render and record as before
        frames = []
        for cam_id in camera_ids:
            frame = model.render(camera_id=cam_id, width=width, height=height)
            frames.append(frame)

        frame_grid = arrange_frames_in_grid(frames, frames_per_row, width, height)
        frame_bgr = cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        model.step()

    # Downward motion
    start_joints_down = model.data.qpos[0:6].copy()
    target_position_down = target_position.copy()
    # Adjust the downward path, this is the final grasping position
    target_position_down[2] = target_position[2] - 0.4  # Adjust the Z-axis to move downward
    target_orientation_euler_down = target_orientation_euler
    target_orientation_down = tf.euler_matrix(*target_orientation_euler_down)[:3, :3]
    joint_angles_down_full = my_chain.inverse_kinematics(target_position_down, target_orientation_down, "all")
    joint_angles_down = joint_angles_down_full[1:7]  # Exclude the base joint if necessary

    # Generate interpolation factors; num specifies the number of interpolations
    num_interpolations = 120
    t_values = np.linspace(0, 1, num=num_interpolations)

    # Generate interpolated joint angle trajectory using linear interpolation
    interpolated_lists_down = []

    for t in t_values:
        # Linear interpolation
        s_t = t
        interpolated_q = (1 - s_t) * start_joints_down + s_t * joint_angles_down
        interpolated_lists_down.append(interpolated_q)

    # Apply the path to pick up motion
    if interpolated_lists_down:
        print("Downward path found")
        open_gripper()

        for q in interpolated_lists_down:
            # Set the joint angles
            model.data.ctrl[0:6] = q  # Control inputs for the joints

            # Render and record as before
            frames = []
            for cam_id in camera_ids:
                frame = model.render(camera_id=cam_id, width=width, height=height)
                frames.append(frame)

            frame_grid = arrange_frames_in_grid(frames, frames_per_row, width, height)
            frame_bgr = cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

            # Record the end-effector pose
            position, orientation = get_end_effector_pose(q)
            orientation_flat = orientation.flatten()
            pose = np.concatenate((position, orientation_flat))
            end_effector_poses.append(pose)

            model.step()

        # Close the gripper to grasp the object
        close_gripper()

        # Wait for a few simulation steps to let the gripper close
        for i in range(30):
            frames = []
            for cam_id in camera_ids:
                frame = model.render(camera_id=cam_id, width=width, height=height)
                frames.append(frame)

            frame_grid = arrange_frames_in_grid(frames, frames_per_row, width, height)
            frame_bgr = cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            model.step()
    else:
        print("No downward path found!")

    # Upward motion
    start_joints_up = joint_angles_down  # Starting from the grasping position
    target_position_up = target_position_down.copy()
    target_position_up[2] = target_position_down[2] + 0.3  # Move up by 0.3 meters

    target_orientation_euler_up = target_orientation_euler
    target_orientation_up = tf.euler_matrix(*target_orientation_euler_up)[:3, :3]
    joint_angles_up_full = my_chain.inverse_kinematics(target_position_up, target_orientation_up, "all")
    joint_angles_up = joint_angles_up_full[1:7]  # Exclude the base joint if necessary

    # Generate interpolated joint angle trajectory for lifting up using S-cubic interpolation
    interpolated_lists_up = []

    for t in t_values:
        # S-cubic interpolation function
        s_t = 3 * t ** 2 - 2 * t ** 3
        interpolated_q = (1 - s_t) * start_joints_up + s_t * joint_angles_up
        interpolated_lists_up.append(interpolated_q)

    if interpolated_lists_up:
        print("Upward path found")
        # Apply the path to the simulation and record the video
        for q in interpolated_lists_up:
            # Set the joint angles
            model.data.ctrl[0:6] = q  # Control inputs for the joints

            # Render and record as before
            frames = []
            for cam_id in camera_ids:
                frame = model.render(camera_id=cam_id, width=width, height=height)
                frames.append(frame)

            frame_grid = arrange_frames_in_grid(frames, frames_per_row, width, height)
            frame_bgr = cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

            # Record the end-effector pose
            position, orientation = get_end_effector_pose(q)
            orientation_flat = orientation.flatten()
            pose = np.concatenate((position, orientation_flat))
            end_effector_poses.append(pose)

            model.step()
    else:
        print("No upward path found!")

    # New Stage: Move to Another Target Position
    # ------------------------------------------

    # Define the new target position (specified by you)
    new_target_position = [0.1, 0.5, 0.25]  # Replace with your desired coordinates
    new_target_orientation_euler = [0, 0, 0]  # Replace with your desired orientation angles
    new_target_orientation = tf.euler_matrix(*new_target_orientation_euler)[:3, :3]

    # Compute joint angles for the new target position using inverse kinematics
    joint_angles_new_full = my_chain.inverse_kinematics(new_target_position, new_target_orientation, "all")
    joint_angles_new = joint_angles_new_full[1:7]  # Exclude the base joint if necessary

    # Start from the current joint configuration (after upward motion)
    start_joints_transfer = joint_angles_up.copy()

    # Generate interpolated joint angle trajectory using S-cubic interpolation
    interpolated_lists_transfer = []

    # Generate interpolation factors
    num_interpolations = 150
    t_values = np.linspace(0, 1, num=num_interpolations)

    for t in t_values:
        # S-cubic interpolation function
        s_t = 3 * t ** 2 - 2 * t ** 3
        interpolated_q = (1 - s_t) * start_joints_transfer + s_t * joint_angles_new
        interpolated_lists_transfer.append(interpolated_q)

    # Apply the transfer motion
    if interpolated_lists_transfer:
        print("Transfer path found")
        for q in interpolated_lists_transfer:
            # Set the joint angles
            model.data.ctrl[0:6] = q  # Control inputs for the joints

            # Render and record as before
            frames = []
            for cam_id in camera_ids:
                frame = model.render(camera_id=cam_id, width=width, height=height)
                frames.append(frame)

            frame_grid = arrange_frames_in_grid(frames, frames_per_row, width, height)
            frame_bgr = cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

            # Record the end-effector pose
            position, orientation = get_end_effector_pose(q)
            orientation_flat = orientation.flatten()
            pose = np.concatenate((position, orientation_flat))
            end_effector_poses.append(pose)

            model.step()
    else:
        print("No transfer path found!")

    # Release the object at the new position
    open_gripper()

    # Wait for a few simulation steps to let the gripper open
    for i in range(50):
        frames = []
        for cam_id in camera_ids:
            frame = model.render(camera_id=cam_id, width=width, height=height)
            frames.append(frame)

        frame_grid = arrange_frames_in_grid(frames, frames_per_row, width, height)
        frame_bgr = cv2.cvtColor(frame_grid, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
        model.step()

    # After the simulation, save the poses to a text file
    np.savetxt(pose_log_file, end_effector_poses, delimiter=',',
               header='px,py,pz,r00,r01,r02,r10,r11,r12,r20,r21,r22')
    print(f"End-effector poses saved to {pose_log_file}")

    # Release the video writer
    out.release()
    print(f"Video saved as {video_name}")


def close_gripper():
    # Gripper close control
    model.data.ctrl[6] = -0.15
    model.data.ctrl[7] = 0.15


def open_gripper():
    # Gripper open control
    model.data.ctrl[6] = 0
    model.data.ctrl[7] = 0


# Example usage:
start = [0., 0., 0., 0., 0., 0.]  # Start joint angles

# Target in Cartesian coordinates based on banana's position
target_position = [0.3, 0.4, 0.55]  # Position of the banana

# Orientation of the gripper to align with the banana
target_orientation_euler = [0, 0, 0]  # Roll=0, Pitch=0°, Yaw=0

# Convert Euler angles to a rotation matrix
target_orientation = tf.euler_matrix(*target_orientation_euler)[:3, :3]

# Inverse Kinematics
joint_angles_full = my_chain.inverse_kinematics(target_position, target_orientation, "all")
joint_angles = joint_angles_full[1:7]  # Exclude the base joint if necessary

# Goal and joint limits
goal = joint_angles
print("goal", goal)
joint_limits = [(-3, 3)] * 6  # Example joint limits
joint_limits[2] = (-3, 0)  # Elbow
joint_limits[3] = (-1.5, 1.5)  # Forearm roll

# Initialize RRT
rrt = RRT(start=start, goal=goal, joint_limits=joint_limits)

# Generate RRT Path
rrt_path = rrt.planning(model)

# Apply the path to the MuJoCo simulation and record video
if rrt_path:
    print("Path found!")

    # Open gripper
    open_gripper()

    # Apply RRT Path
    apply_rrt_path_to_dm_control(model, rrt_path, video_name="rrt_robot_motion_with_transfer.mp4",
                                 pose_log_file="end_effector_poses_with_transfer.txt")
else:
    print("No path found!")

    # Still save a video to visualize the failure
    # Create a dummy path with the start configuration
    dummy_path = [start]

    # Apply the dummy path to the simulation and record video
    apply_rrt_path_to_dm_control(model, dummy_path, video_name="rrt_robot_motion_failure.mp4",
                                 pose_log_file="end_effector_poses_failure.txt")
