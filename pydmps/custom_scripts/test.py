import matplotlib.pyplot as plt
import time

# Sample data: replace these lists with your own x and y coordinates
x_coordinates = [1, 2, 3, 4, 5]
y_coordinates = [2, 4, 1, 3, 5]

# Plot the path point by point with a delay
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

for x, y in zip(x_coordinates, y_coordinates):
    ax.plot(x, y, marker='o', linestyle='', color='b')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Path Plot')
    plt.draw()
    plt.pause(1)  # Adjust the pause duration in seconds

# Keep the plot window open
plt.ioff()
plt.show()
