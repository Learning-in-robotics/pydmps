import numpy as np
import yaml

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

    # Create a list of [x, y] values
    xy_values = [[float(x_val), float(y_val)] for x_val, y_val in zip(x, y_combined)]

    return xy_values


xy_values = generate_random_curve()
data_to_save = {
    'trajectory':  np.asarray(xy_values).tolist()
}
with open('/home/hamsadatta/test/dmp/my/pydmps/utils/input_data.yaml', 'w') as file:
    yaml.dump(data_to_save, file)
