import os
import yaml
import numpy as np
from pydmps.dmp_discrete import DMPs_discrete

current_directory = os.path.dirname(os.path.abspath(__file__))

class DMPDataGenerator:
    def __init__(self, config_path):
        self.config = self.read_yaml_file(config_path)
        self.n_dmps = self.config['n_dmps']
        self.n_bfs= self.config['n_bfs']
        self.input_path= current_directory + self.config['input_path']
        self.output_path= current_directory + self.config['output_path']
        self.input_data =  np.array(self.read_yaml_file(self.input_path)['trajectory'])

    def read_yaml_file(self, path):
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def write_yaml_file(self, output_path, data):
        with open(output_path, 'w') as file:
            yaml.dump(data, file)

    def process_dmp(self, input_data):
        dmp = DMPs_discrete(n_dmps=self.n_dmps, n_bfs=self.n_bfs)
        dmp.imitate_path(y_des=input_data)
        dmp.goal = [trajectory[-1] for trajectory in input_data]
        return dmp.rollout()

    def run(self):        
        x_path = self.input_data[:, 0]
        y_path = self.input_data[:, 1]

        y_track, dy_track, ddy_track, clock_track, f_track = self.process_dmp(np.array([x_path, y_path]))
        output_data = {
            'y_track': y_track.tolist(),
            'dy_track': dy_track.tolist(),
            'ddy_track': ddy_track.tolist(),
            'clock_track': clock_track,
            'f_track': f_track.tolist()
        }
        self.write_yaml_file(self.output_path, output_data)

if __name__ == "__main__":
    dmp_data = DMPDataGenerator(config_path=current_directory + '/config.yaml')
    dmp_data.run()
