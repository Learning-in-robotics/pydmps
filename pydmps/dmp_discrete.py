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
        self.num_epochs = 10000

        

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
 
    plt.figure(1, figsize=(6, 3))
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
    plt.figure(2, figsize=(6, 4))
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


    # for ii, bfs in enumerate(n_bfs):
    dmp = DMPs_discrete(n_dmps=2, n_bfs=0)

    dmp.imitate_path(y_des=np.array([path1, path2]))
    # change the scale of the movement
    dmp.goal[0] = 0.5
    dmp.goal[1] = -2.5

    y_track, dy_track, ddy_track, f_track = dmp.rollout(current_point)




    # plt.figure(2)
    # plt.subplot(211)
    # plt.plot(y_track[:, 0], lw=2)
    # plt.subplot(212)
    # plt.plot(y_track[:, 1], lw=2)
    # plt.subplot(213)
    # plt.plot(f_values, label="Forcing Terms")
    # print(f_track)

    # plt.subplot(211)
    # a = plt.plot(path1 / path1[-1] * dmp.goal[0], "r--", lw=2)
    # plt.title("DMP imitate path")
    # plt.xlabel("time (ms)")
    # plt.ylabel("system trajectory")
    # plt.legend([a[0]], ["desired path"], loc="lower right")
    # plt.subplot(212)
    # b = plt.plot(path2 / path2[-1] * dmp.goal[1], "r--", lw=2)
    # plt.title("DMP imitate path")
    # plt.xlabel("time (ms)")
    # plt.ylabel("system trajectory")
    # plt.legend(["%i BFs" % i for i in n_bfs], loc="lower right")

    # plt.tight_layout()


    # plt.figure(3)
    # plt.scatter(y_track[:, 0], y_track[:, 1], linestyle='-')

    # plt.figure(4)
    # plt.scatter(f_track[:, 0], f_track[:, 1], linestyle='-')



    # plt.show()
