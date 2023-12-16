import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import lstsq

class DiscreteDMP(object):
    def __init__(self, n_basis_functions=1000):
        # YOUR CODE HERE
        
        self.alpha = 25.0
        self.beta = self.alpha/4
        self.tau = 1
        self.ax = 1.0
         # Number of basis functions
        self.n_basis_functions = n_basis_functions
        # Centers and widths of the Gaussian basis functions
        self.centers = np.linspace(0, 1, n_basis_functions)
        self.widths = np.ones(n_basis_functions) * (0.5 / (n_basis_functions - 1))**2
        # Weights for the basis functions, initialized to zero
        self.w = np.zeros((n_basis_functions, 3))
        self.y0 = None
        self.g = None
    
    def basis_function(self, x,i):
        # Gaussian basis function
        return np.exp(-0.5 * (x - self.centers[i])**2 / self.widths[i])

    def forcing_term(self, x, d):
        # Forcing term is a weighted sum of the basis functions
        psi = np.array([self.basis_function(x, i) for i in range(self.n_basis_functions)])
        zeta = x * (self.g - self.y0)
        return  np.dot(psi, self.w[:, d]) / np.sum(psi)

    def learn(self,path) -> None:
        """Learns a dynamic motion primitive.
        """
        # YOUR CODE HERE
        assert path.shape[1] == 3, "Path must be three-dimensional"
        
        self.y0 = path[0]
        self.g = path[-1]
        n_steps = len(path)
        dt = 1.0 / n_steps
        x = 1
        self.tau =1

        
        x_can=np.zeros(n_steps)
        for t in range(n_steps):
            x = x + (-self.ax * x ) * self.tau * dt
            x_can[t] = x
            




        f_target = np.zeros((n_steps, 3))
        for d in range(3):
            if abs(self.g[d] - self.y0[d]) < np.finfo(float).eps:
                f_target[:, d] = np.zeros(n_steps)
            else:
                f_target[:, d] = (path[:, d] - self.y0[d]) / (self.g[d] - self.y0[d]) - 1
                
            psi_matrix = np.array([[self.basis_function(x[t], i) for i in range(self.n_basis_functions)] for t in range(n_steps)])
            self.w[:, d] = lstsq(psi_matrix, f_target[:, d], rcond=None)[0]
            
    def unroll(self, y0: np.ndarray, g: np.ndarray, n_steps=100) -> np.ndarray:
        """Generates a path that starts at y0 and ends at g,
        following a learned DMP shape.

        Keyword arguments:
        y0: np.ndarray -- Initial state of the system
        g: np.ndarray -- Goal state

        Returns a numpy array representing path waypoints.

        """
        # YOUR CODE HERE
        y = np.zeros((n_steps, 3))
        y[0] = y0
        dt = 1.0 / n_steps
        
        for t in range(1, n_steps):
            x = np.exp(-self.alpha * t * dt / self.tau)
            for d in range(3):  
                f_d = self.forcing_term(x, d)
                yddot_d = self.alpha * (self.beta * (g[d] - y[t-1, d]) - dt * f_d)
                y[t, d] = y[t-1, d] + dt * yddot_d
                
        return y


def generate_line(n_steps=10):
    """Generates a straight line in 3D."""
    t = np.linspace(0, 1, n_steps)
    return np.array([t, t, t]).T

def generate_demonstration_points(a=1.0, b=1.0, c=1.0, x_range=(0, 5), num_points=100, height=0.5, visualize=False):
    # Create an array of values for the x-coordinate (only the positive half)
    x = np.linspace(x_range[0], x_range[1], num_points)

    # Calculate the corresponding y and z values for the upper half of the parabolic trajectory
    y = height * x
    z = a * x**2 + b * y**2 + c

    points = np.array([x, y, z]).T

    if visualize:
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the upper half of the parabolic trajectory
        ax.plot(x, y, z)

        # Set labels for the axes
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        
    return points

def evaluate(dmp, curve):
    """Evaluates the reconstruction error of a DMP."""
    y0 = curve[0]
    g = curve[-1]
    reconstructed_curve = dmp.unroll(y0, g, len(curve))
    error = np.mean((curve - reconstructed_curve)**2)
    return reconstructed_curve, error

def main():
    # Generate a curve
    curve = generate_demonstration_points()

    # Learn and unroll a DMP
    dmp = DiscreteDMP(n_basis_functions=100)
    dmp.learn(curve)
    y0 = curve[0]
    g = curve[-1]
    reconstructed_curve = dmp.unroll(y0, g, len(curve))

    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], label='Original')  # Plot the original curve in 3D
    ax.plot(reconstructed_curve[:, 0], reconstructed_curve[:, 1], reconstructed_curve[:, 2], label='Reconstructed')  # Plot the reconstructed curve in 3D
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()