import os
import yaml
import numpy as np

class TrajectoryParser:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.data_matrix = []
        self.labels_matrix = []

    def parse_trajectory_file(self, file_path):
        with open(file_path, 'r') as file:
            trajectory_data = yaml.safe_load(file)

        y_track_data = trajectory_data.get('y_track', [])
        f_track_data = trajectory_data.get('f_track', [])
        clock_track_data = trajectory_data.get('clock_track', [])

        return [y_track_data[i] + [clock_track_data[i]] for i in range(len(y_track_data))], f_track_data

    def process_folder(self):        

        file_list = sorted(os.listdir(self.folder_path), key=lambda x: int(x.split('_')[1].split('.')[0]))

        for filename in file_list:
            if filename.startswith("trajectory_") and filename.endswith(".yaml"):
                file_path = os.path.join(self.folder_path, filename)
                data, labels = self.parse_trajectory_file(file_path)
                self.data_matrix.append(data)
                self.labels_matrix.append(labels)

        self.data_matrix = np.array(self.data_matrix)
        self.labels_matrix = np.array(self.labels_matrix)

# # Example usage:
# # Replace 'your_folder_path' with the actual path to the folder containing trajectory files
# folder_path = '/home/hamsadatta/test/dmp/my/pydmps/utils/dataset'
# parser = TrajectoryParser(folder_path)
# parser.process_folder()

# data_matrix = parser.data_matrix
# labels_matrix = parser.labels_matrix

# print("Data Matrix Shape:", len(data_matrix))
# print("Labels Matrix Shape:", len(labels_matrix))
