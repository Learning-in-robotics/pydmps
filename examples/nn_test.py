import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        
        # Define the layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Define activation function (you can change this to another activation if needed)
        self.activation = nn.ReLU()

    def forward(self, x):
        # Forward pass through the network
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.hidden_layer1(x)
        x = self.activation(x)
        x = self.hidden_layer2(x)
        x = self.activation(x)
        x = self.output_layer(x)
        
        return x

# Specify the input, hidden, and output layer sizes
input_size = 2  # Input layer accepts (x, y)
hidden_size = 64  # Number of neurons in each hidden layer
output_size = 2  # Output layer predicts forcing terms for (x, y)

# Create an instance of the neural network
net = NeuralNetwork(input_size, hidden_size, output_size)

# You can use this network to make predictions like this:
input_data = torch.Tensor([1, 2])  # Replace x and y with actual values
forcing_terms = net(input_data)

# Print the predicted forcing terms
print(forcing_terms)
