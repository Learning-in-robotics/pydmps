import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read YAML file
file_path = '/home/chaitanya/RL project/pydmps/pydmps/utils/trajectories/recorded_poses_1.yaml'

with open(file_path, 'r') as file:
    data = yaml.safe_load(file)

# Extract trajectory data
trajectory = data['trajectory']

# Extract x, y, z coordinates
x = [pose[0] for pose in trajectory]
y = [pose[1] for pose in trajectory]
z = [pose[2] for pose in trajectory]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory in 3D
ax.plot(x, y, z, marker='o', linestyle='-', label='Trajectory')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Trajectory Plot')

# Set viewing perspective
ax.view_init(elev=90, azim=-90)

# Invert axes
# ax.invert_xaxis()
ax.invert_yaxis()
ax.invert_zaxis()

# Show the plot
plt.show()
