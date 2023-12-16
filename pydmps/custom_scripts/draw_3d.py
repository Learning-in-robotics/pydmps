"""
Copyright (C) 2016 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import matplotlib.pyplot as plt

import pydmps
import pydmps.dmp_discrete

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

        # Show the plot
        plt.show()

    return points

def plot_trajectories(actual_points, demo_points, actual_label='Actual', demo_label='Demo'):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the actual trajectory with a solid line
    ax.plot(actual_points[:, 0], actual_points[:, 1], actual_points[:, 2], label=actual_label)

    # Plot the demo trajectory with dashed lines
    ax.plot(demo_points[:, 0], demo_points[:, 1], demo_points[:, 2], 'r--', label=demo_label)

    # Set labels for the axes
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()


# Example usage:
points = generate_demonstration_points(num_points=79)

y_des = generate_demonstration_points().T

y_des -= y_des[:, 0][:, None]



# test normal run
dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=3, n_bfs=500)
y_track = []
dy_track = []
ddy_track = []

dmp.imitate_path(y_des=y_des, plot=False)
y_track, dy_track, ddy_track = dmp.rollout()

plot_trajectories(y_track,y_des)


