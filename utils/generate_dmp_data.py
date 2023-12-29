import os
import yaml
import re
import numpy as np
from pydmps.dmp_discrete import DMPs_discrete

current_directory = os.path.dirname(os.path.abspath(__file__))

class DMPDataGenerator:
    def __init__(self, config_path):
        self.config = self.read_yaml_file(config_path)
        self.n_dmps = self.config['n_dmps']
        self.n_bfs= self.config['n_bfs']
        self.input_folder= current_directory + self.config['input_folder']
        self.output_folder= current_directory + self.config['output_folder']

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
        input_files = os.listdir(self.input_folder)
        for _, input_file in enumerate(input_files):
            match = re.search(r'\d+', input_file)
            if match:
                index = int(match.group())
                print(index)
            else:
                print("Please check the filename.")
            input_path = os.path.join(self.input_folder, input_file)
            output_path = os.path.join(self.output_folder, f'trajectory_{index}.yaml')

            input_data = np.array(self.read_yaml_file(input_path)['trajectory'])
            x_path = input_data[:, 0]
            y_path = input_data[:, 1]
            z_path = input_data[:, 2]
            roll_path = input_data[:, 3]
            pitch_path = input_data[:, 4]
            yaw_path = input_data[:, 5]

            y_track, dy_track, ddy_track, clock_track, f_track = self.process_dmp(np.array([x_path, y_path, z_path,
                                                                                            roll_path, pitch_path, yaw_path]))
            output_data = {
                'y_track': y_track.tolist(),
                'dy_track': dy_track.tolist(),
                'ddy_track': ddy_track.tolist(),
                'clock_track': clock_track,
                'f_track': f_track.tolist()
            }
            self.write_yaml_file(output_path, output_data)



if __name__ == "__main__":
    dmp_data = DMPDataGenerator(config_path=current_directory + '/config.yaml')
    dmp_data.run()
