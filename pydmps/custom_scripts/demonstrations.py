import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_demonstration_points(a=1.0, b=1.0, c=1.0, x_range=(0, 5), num_points=100, height=0.5, visualize=False):
    # Create an array of values for the x-coordinate (only the positive half)
    x = np.linspace(x_range[0], x_range[1], num_points)

    # Calculate the corresponding y and z values for the upper half of the parabolic trajectory
    y = height * x
    z = a * x**2 + b * y**2 + c

    points = np.array([x, y, z]).T

    if visualize:
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the upper half of the parabolic trajectory
        ax.plot(x, y, z)

        # Set labels for the axes
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        
    return points

# Example usage:
points = generate_demonstration_points(visualize=True)
x_points = points[:, 0]
y_points = points[:, 1]
z_points = points[:, 2]

plt.show()


