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

from pydmps.cs import CanonicalSystem
from pydmps.utils.dmpnet import DMPNetwork
from pydmps.utils.parser import TrajectoryParser


class DMPs(object):
    """Implementation of Dynamic Motor Primitives,
    as described in Dr. Stefan Schaal's (2002) paper."""

    def __init__(
        self, n_dmps, load_model = False, dt=0.01, y0=0, goal=1, ay=None, by=None, **kwargs
    ):
        """
        n_dmps int: number of dynamic motor primitives
        n_bfs int: number of basis functions per DMP
        dt float: timestep for simulation
        y0 list: initial state of DMPs
        goal list: goal state of DMPs
        w list: tunable parameters, control amplitude of basis functions
        ay int: gain on attractor term y dynamics
        by int: gain on attractor term y dynamics
        """

        self.n_dmps = n_dmps
        self.dt = dt
        if isinstance(y0, (int, float)):
            y0 = np.ones(self.n_dmps) * y0
        self.y0 = y0
        if isinstance(goal, (int, float)):
            goal = np.ones(self.n_dmps) * goal
        self.goal = goal
        self.dataset_path = '/home/hamsadatta/test/dmp/rl/pydmps/pydmps/utils/dataset'
        self.save_model_path = '/home/hamsadatta/test/dmp/rl/pydmps/pydmps/utils/trained_model.pt'        
        self.net = DMPNetwork(7, 128, 6)

        if load_model == True:             
            # Load the trained model's state dictionary
            self.net.load_state_dict(torch.load(self.save_model_path))
            # Set the model to evaluation mode
            self.net.eval()
        

        self.ay = np.ones(n_dmps) * 25.0 if ay is None else ay  # Schaal 2012
        self.by = self.ay / 4.0 if by is None else by  # Schaal 2012

        # set up the CS
        self.cs = CanonicalSystem(dt=self.dt, **kwargs)
        self.timesteps = int(self.cs.run_time / self.dt)

        # set up the DMP system
        self.reset_state()

    def check_offset(self):
        """Check to see if initial position and goal are the same
        if they are, offset slightly so that the forcing term is not 0"""

        for d in range(self.n_dmps):
            if abs(self.y0[d] - self.goal[d]) < 1e-4:
                self.goal[d] += 1e-4
    
    def train_network(self):

        print("Data Loading...")
        parser = TrajectoryParser(self.dataset_path)
        parser.process_folder()
        print("Data Loading Done")

        data = parser.data_matrix
        labels = parser.labels_matrix
        num_of_trajectories = len(data)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

        print("Starting Training....")

        # Training loop
        for epoch in range(self.num_epochs):
            total_loss = 0.0  # Initialize the total loss for this epoch

            # Iterate through each trajectory
            for i in range(num_of_trajectories):
                # Forward pass
                outputs = self.net(torch.Tensor(data[i]))
                loss = criterion(outputs, torch.Tensor(labels[i]))
                
                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()  # Accumulate the loss for this batch
            
            # Print training progress
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {total_loss:.4f}')

        # Save the trained model to a .pt file
        torch.save(self.net.state_dict(), self.save_model_path)

        print("Training finished. Model saved as 'trained_model.pt'.")
    
    def get_forcing_term(self,input_data):
        with torch.no_grad():
            input_data = torch.Tensor(input_data)
            f_term = self.net(input_data)

        return f_term

    def imitate_path(self):

        self.train_network()

        self.reset_state()


    def rollout(self, current_point, timesteps=1.0, **kwargs): 
        """Generate a system trial, no feedback is incorporated."""

        # self.reset_state()

        # if timesteps is None:
        #     if "tau" in kwargs:
        #         timesteps = int(self.timesteps / kwargs["tau"])
        #     else:
        #         timesteps = self.timesteps

        # set up tracking vectors
        y_track = np.zeros((timesteps, self.n_dmps))
        dy_track = np.zeros((timesteps, self.n_dmps))
        ddy_track = np.zeros((timesteps, self.n_dmps))
        f_track = np.zeros((timesteps, self.n_dmps))


        # for t in range(timesteps):
            # run and record timestep
        y_track, dy_track, ddy_track, f_track, x  = self.step(current_point, **kwargs)
            # clock_track.append(x)

        return y_track, dy_track, ddy_track, f_track, x

    def reset_state(self):
        """Reset the system state"""
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.f_val = np.zeros(self.n_dmps)
        self.cs.reset_state()

    def step(self, current_point, tau=1.0, error=0.0, external_force=None):
        """Run the DMP system for a single timestep.

        tau float: scales the timestep
                   increase tau to make the system execute faster
        error float: optional system feedback
        """
        error_coupling = 1.0 / (1.0 + error)
        # run canonical system
        x = self.cs.step(tau=tau, error_coupling=error_coupling)
        current_point.append(x)
        
        # generate the forcing term
        f = self.get_forcing_term(current_point)  

        for d in range(self.n_dmps):
            self.f_val[d]= f[d]            
            # DMP acceleration
            self.ddy[d] = (
                self.ay[d] * (self.by[d] * (self.goal[d] - self.y[d]) - self.dy[d]) + f[d]
            )
            if external_force is not None:
                self.ddy[d] += external_force[d]
            self.dy[d] += self.ddy[d] * tau * self.dt * error_coupling
            self.y[d] += self.dy[d] * tau * self.dt * error_coupling

        return self.y, self.dy, self.ddy, f ,x
