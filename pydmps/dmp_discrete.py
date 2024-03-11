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

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from pydmps.dmp import DMPs
from pydmps.utils.dmpnet import DMPNetwork
from pydmps.utils.parser import TrajectoryParser
from pydmps.utils.plot_utils import plot_2d, plot_pose

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy


class DMPs_discrete(DMPs):
    """An implementation of discrete DMPs"""

    def __init__(self, load_model=False, current_point=None, goal=None, model_name=None, **kwargs):

        # Specify the input, hidden, and output layer sizes
        self.input_size = 7  # Input layer accepts (x, y)
        self.hidden_size = 128  # Number of neurons in each hidden layer
        self.output_size = 6  # Output layer predicts forcing terms for (x, y)
        self.learning_rate = 0.01
        self.num_epochs = 5000
        self.batch_size = 100

        # call super class constructor
        super(DMPs_discrete, self).__init__(
            pattern="discrete",
            load_model=load_model,
            y0=current_point,
            goal=goal,
            model_name=model_name,
            **kwargs,
        )

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
    argparse = argparse.ArgumentParser(description="DMPs_discrete")
    argparse.add_argument("--train", action="store_true", help="Train the network")
    argparse.add_argument("--test", action="store_true", help="Test the network")
    argparse.add_argument("--model_name", type=str, help="Name of the model to save")
    argparse.add_argument("--plot_3d", action="store_true", help="Plot in 3D")

    args = argparse.parse_args()

    train_network = args.train
    test_network = args.test
    plot_3d = args.plot_3d

    if train_network:
        load_model = False
    else:
        load_model = True

    if args.model_name is None and not train_network:
        model_name = "best"
    else:
        model_name = args.model_name if args.model_name is not None else "trained_model"

    current_point = [
        -0.22413162636020156,
        0.03672257898841529,
        1.0070652249653813,
        -2.672549100764965,
        0.11178945152440055,
        -0.05323406781316968,
    ]

    my_goal = [
        0.2405904497616017,
        0.025670480673161248,
        0.9405305042288513,
        -2.627904589695221,
        0.07141495502943067,
        -0.08656735603506686,
    ]

    dmp = DMPs_discrete(
        n_dmps=6,
        load_model=load_model,
        current_point=current_point,
        goal=my_goal,
        model_name=model_name,
    )

    if train_network:
        dmp.imitate_path()
        exit()

    # change the scale of the movement
    dmp.goal = my_goal
    # dmp.goal[1] = -2.51
    test = np.array([])
    # add current_point to test
    test = np.append(test, current_point)
    x_track = 0.99
    print("start")
    while x_track > 0.0001:
        y_track, dy_track, ddy_track, f_track, x_track = dmp.rollout(current_point, 1)
        current_point = copy.deepcopy(y_track)
        # append the y_track to test
        test = np.vstack((test, y_track))
    print("end")
    
    # convert to numpy array
    test = np.array(test)

    # load trajectory_1.yaml from dataset
    dataset_path = "pydmps/dataset"
    # load yaml file
    with open(f"{dataset_path}/trajectory_1.yaml") as file:
        trajectory1 = yaml.load(file, Loader=yaml.FullLoader)

    trajectory1_y_track = trajectory1["y_track"]
    trajectory1_y_track = np.array(trajectory1_y_track)

    if plot_3d:
        plot_pose(test, trajectory1_y_track)
    else:
        # plot xy, yz and zx points
        plot_2d(test, trajectory1_y_track)
