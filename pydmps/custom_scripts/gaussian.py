import numpy as np

def gaussian_basis(x, center, width):
    """
    Calculate the values of a Gaussian basis function at the given data points x.

    Parameters:
    - x: Input data points (array or scalar)
    - center: Center of the Gaussian basis function
    - width: Width (standard deviation) of the Gaussian basis function

    Returns:
    - Values of the Gaussian basis function for each data point x
    """
    return np.exp(-0.5 * ((x - center) / width) ** 2)

# Example usage:
x = np.linspace(-5, 5, 100)  # Generate some sample data points
centers = np.array([1])  # Gaussian centers
width = 1.0  # Width (standard deviation)

# Calculate the values of Gaussian basis functions for each data point and each center
basis_functions = np.vstack([gaussian_basis(x, center, width) for center in centers])

print(basis_functions[0])
# Plot the Gaussian basis functions
import matplotlib.pyplot as plt

for i, center in enumerate(centers):
    plt.plot(x, basis_functions[i], label=f'Center at {center}')

plt.title('Gaussian Basis Functions')
plt.legend()
plt.xlabel('x')
plt.ylabel('Basis Function Value')
plt.show()
