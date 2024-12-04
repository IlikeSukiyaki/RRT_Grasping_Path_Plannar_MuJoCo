# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load the calculation-based end-effector poses
calc_pose_log_file = "./end_effector_poses_with_transfer_calculation.txt"
data_calc = np.loadtxt(calc_pose_log_file, delimiter=',', skiprows=1)  # Skip header row

# Load the simulation-based end-effector poses
sim_pose_log_file = "./end_effector_poses_with_transfer_simulation.txt"
data_sim = np.loadtxt(sim_pose_log_file, delimiter=',', skiprows=1)  # Skip header row

# Extract position data for calculation-based trajectory
positions_calc = data_calc[:, :3]  # Columns px, py, pz
px_calc, py_calc, pz_calc = positions_calc[:, 0], positions_calc[:, 1], positions_calc[:, 2]

# Extract position data for simulation-based trajectory
positions_sim = data_sim[:, :3]  # Columns px, py, pz
px_sim, py_sim, pz_sim = positions_sim[:, 0], positions_sim[:, 1], positions_sim[:, 2]

# Plot the trajectories
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the calculation-based trajectory as a solid line
ax.plot(px_calc, py_calc, pz_calc, label='Theoretical RRT Trajectory', color='blue', linewidth=2)

# Plot the simulation-based trajectory as a dashed line
ax.plot(px_sim, py_sim, pz_sim, label='Simulated RRT Trajectory', color='red', linewidth=2, linestyle='--')

# Optionally, plot the start and end points
ax.scatter(px_calc[0], py_calc[0], pz_calc[0], color='green', marker='o', s=100, label='Start Point')
ax.scatter(px_calc[-1], py_calc[-1], pz_calc[-1], color='blue', marker='X', s=100, label='End Point (Theo)')
ax.scatter(px_sim[-1], py_sim[-1], pz_sim[-1], color='red', marker='X', s=100, label='End Point (Sim)')

# Set labels and title with increased font sizes
ax.set_xlabel('X Position (m)', fontsize=14, fontweight='bold')
ax.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold')
ax.set_zlabel('Z Position (m)', fontsize=14, fontweight='bold')
ax.set_title('End-Effector Trajectories Comparison', fontsize=16, fontweight='bold')

# Set axis limits explicitly
x_min = min(px_calc.min(), px_sim.min()) - 0.1
x_max = max(px_calc.max(), px_sim.max()) + 0.1
y_min = min(py_calc.min(), py_sim.min()) - 0.1
y_max = max(py_calc.max(), py_sim.max()) + 0.1
z_min = min(pz_calc.min(), pz_sim.min()) - 0.1
z_max = max(pz_calc.max(), pz_sim.max()) + 0.1
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

# Adjust tick parameters for better visibility
ax.tick_params(axis='both', which='major', labelsize=12, width=2)
ax.tick_params(axis='both', which='minor', labelsize=10)

# Add grid lines with increased line width
ax.grid(True, linewidth=0.5)

# Elongate the x-axis by adjusting the aspect ratio
x_range = x_max - x_min
y_range = y_max - y_min
z_range = z_max - z_min

# Multiply the x_range to elongate the x-axis
ax.set_box_aspect([x_range * 3, y_range, z_range])  # Adjust the multiplier as needed

# Optionally, set the viewing angle
ax.view_init(elev=30, azim=60)  # Adjust the elevation and azimuth as needed

# Add legend with increased font size
ax.legend(fontsize=12)

# Show the plot
plt.show()
