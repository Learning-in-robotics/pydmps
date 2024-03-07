"""
Copyright (C) 2013 Travis DeWolf

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
import torch
import torch.nn as nn
import torch.optim as optim

from pydmps.dmp import DMPs
from pydmps.utils.dmpnet import DMPNetwork
from pydmps.utils.parser import TrajectoryParser

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy

def plot_pose(y_tracks):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for y_track in y_tracks:
        ax.scatter(y_track[0], y_track[1], y_track[2])
    
    # ax.plot(x, y, z)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

class DMPs_discrete(DMPs):
    """An implementation of discrete DMPs"""

    def __init__(self, **kwargs):
        # call super class constructor
        super(DMPs_discrete, self).__init__(pattern="discrete", **kwargs)

         

        # Specify the input, hidden, and output layer sizes
        self.input_size = 7  # Input layer accepts (x, y)
        self.hidden_size = 128  # Number of neurons in each hidden layer
        self.output_size = 6  # Output layer predicts forcing terms for (x, y)
        self.learning_rate = 0.01
        self.num_epochs = 1000

        

        self.check_offset()

    
   

    def gen_front_term(self, x, dmp_num):
        """Generates the diminishing front term on
        the forcing term.

        x float: the current value of the canonical system
        dmp_num int: the index of the current dmp
        """
        return x * (self.goal[dmp_num] - self.y0[dmp_num])

    def gen_goal(self, y_des):
        """Generate the goal for path imitation.
        For rhythmic DMPs the goal is the average of the
        desired trajectory.

        y_des np.array: the desired trajectory to follow
        """

        return np.copy(y_des[:, -1])
    


# ==============================
# Test code
# ==============================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # test normal run
    # dmp = DMPs_discrete(dt=0.05, n_dmps=1, n_bfs=10, w=np.zeros((1, 10)))
    # y_track, dy_track, ddy_track, f_track  = dmp.rollout()
 
    # plt.figure(1, figsize=(6, 3))
    # plt.plot(np.ones(len(y_track)) * dmp.goal, "r--", lw=2)
    # plt.plot(y_track, lw=2)
    # plt.title("DMP system - no forcing term")
    # plt.xlabel("time (ms)")
    # plt.ylabel("system trajectory")
    # plt.legend(["goal", "system state"], loc="lower right")
    # plt.tight_layout()

    def generate_random_curve(start_point=(0, 0), end_point=(5, 3), amplitude_sin=0.5, frequency_sin=2*np.pi/(5-0), phase_shift_sin=np.pi/2):
        # Generate x points between start_point and end_point
        x = np.linspace(start_point[0], end_point[0], 100)

        # Fit the inverted parabola to the given points (0, 0) and (5, 3)
        a_parabola = -end_point[1] / end_point[0]**2
        b_parabola = 0
        c_parabola = 0

        # Calculate y points using the inverted parabola
        y_parabola = a_parabola * x**2 + b_parabola * x + c_parabola

        # Calculate y points for the sine curve
        y_sin = amplitude_sin * np.sin(frequency_sin * (x - start_point[0]) + phase_shift_sin)

        # Combine the parabola and sine curve
        y_combined = y_parabola + y_sin

        # Return (x, y) as NumPy arrays
        return np.array([x, y_combined])

    xy_array = generate_random_curve()

    # test imitation of path run
    # plt.figure(2, figsize=(6, 4))
    # n_bfs = [10, 30, 50, 100, 10000]
    n_bfs = [100]

    # a straight line to target
    path1 = np.sin(np.arange(0, 1, 0.01) * 5)
    # a strange path to target
    path2 = np.zeros(path1.shape)
    path2[int(len(path2) / 2.0) :] = 0.5

    #comment below when not using the random curve
    path1 = xy_array[0]
    path2 = xy_array[1]

    current_point = [-0.22413162636020156, 0.03672257898841529, 1.0070652249653813, -2.672549100764965, 0.11178945152440055, -0.05323406781316968]

    my_goal = [0.2405904497616017, 0.025670480673161248, 0.9405305042288513, -2.627904589695221, 0.07141495502943067, -0.08656735603506686]
    # for ii, bfs in enumerate(n_bfs):
    dmp = DMPs_discrete(n_dmps=6)

    # dmp.imitate_path()
    # change the scale of the movement
    dmp.goal = my_goal
    # dmp.goal[1] = -2.51
    test=[]
    x_track = 0.99
    print("start")
    while (x_track>0.0001):
        y_track, dy_track, ddy_track, f_track, x_track = dmp.rollout(current_point,1)
        print(y_track)        
        current_point = copy.deepcopy(y_track.tolist())
        test.append(current_point)        
    # test = np.array(test)
    print(test)
    plot_pose(test)
    