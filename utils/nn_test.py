import torch
import torch.nn as nn
import torch.optim as optim

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


data = torch.rand((100, 7))
labels = torch.rand((100, 6))

# Specify the input, hidden, and output layer sizes
input_size = 7  # Input layer accepts (x, y)
hidden_size = 128  # Number of neurons in each hidden layer
output_size = 6  # Output layer predicts forcing terms for (x, y)
learning_rate = 0.01
num_epochs = 10000

# Create an instance of the neural network
net = NeuralNetwork(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = net(data)
    loss = criterion(outputs, labels)
    
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print training progress
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model to a .pt file
torch.save(net.state_dict(), 'trained_model.pt')

print("Training finished. Model saved as 'trained_model.pt'.")








# # You can use this network to make predictions like this:
# input_data = torch.Tensor([1, 2])  # Replace x and y with actual values
# forcing_terms = net(input_data)

# # Print the predicted forcing terms
# print(forcing_terms)
