import numpy as np
import matplotlib.pyplot as plt

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

# Call the function and get the (x, y) array
xy_array = generate_random_curve()

# Plot the result
plt.scatter(xy_array[0], xy_array[1], marker='o', color='b')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Combined Function Plot')
plt.show()

# Print or use xy_array as needed
print("xy array:")
print(xy_array)

