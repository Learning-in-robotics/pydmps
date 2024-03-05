import torch
import torch.nn as nn
import torch.optim as optim
from pydmps.utils.parser import TrajectoryParser

class DMPNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DMPNetwork, self).__init__()
        
        # Define the layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Activation function 
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



if __name__ == "__main__":

    folder_path = '/home/hamsadatta/test/dmp/my/pydmps/pydmps/utils/dataset'
    print("Data Loading...")
    parser = TrajectoryParser(folder_path)
    parser.process_folder()
    print("Data Loading Done")

    data = parser.data_matrix
    labels = parser.labels_matrix

    # Specify the input, hidden, and output layer sizes
    input_size = 7  # Input layer accepts (x, y)
    hidden_size = 128  # Number of neurons in each hidden layer
    output_size = 6  # Output layer predicts forcing terms for (x, y)
    learning_rate = 0.01
    num_epochs = 10000
    num_of_trajectories = len(data)

    # Create an instance of the neural network
    net = DMPNetwork(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    print("Starting Training....")


    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0  # Initialize the total loss for this epoch

        # Iterate through each trajectory
        for i in range(num_of_trajectories):
            # Forward pass
            outputs = net(torch.Tensor(data[i]))
            loss = criterion(outputs, torch.Tensor(labels[i]))
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # Accumulate the loss for this batch
        
        # Print training progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}')

    # Save the trained model to a .pt file
    torch.save(net.state_dict(), 'trained_model.pt')

    print("Training finished. Model saved as 'trained_model.pt'.")








    # # You can use this network to make predictions like this:
    # input_data = torch.Tensor([1, 2])  # Replace x and y with actual values
    # forcing_terms = net(input_data)

    # # Print the predicted forcing terms
    # print(forcing_terms)
