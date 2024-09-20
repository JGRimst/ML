import numpy as np
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time


import torch
import torch.nn as nn 

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")



class JordanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation = nn.ReLU(), loss_eq = nn.MSELoss(), optimizer = torch.optim.Adam, learning_rate=0.001, device=None):
        super(JordanRNN, self).__init__()

        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + output_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        
        self.loss_ = loss_eq
        self.optimizer = optimizer

        self.activation = activation
        self.to(self.device)

        self.optimizer = optimizer(self.parameters(), lr=learning_rate)

    def forward(self, input, prev_output):

        input = input.to(self.device)
        prev_output = prev_output.to(self.device)

        if input.dim() == 1:
            input = input.unsqueeze(0)  # Shape: [1, input_size]
        
        if prev_output.dim() == 1:
            prev_output = prev_output.unsqueeze(0)  # Shape: [1, output_size]


        combined = torch.cat((input, prev_output), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.activation(output)
        return output
    
    def init_output(self):
        return torch.zeros(1, self.output_size, device=self.device)
    
    def fit(self, X, y, epochs=100):
        """
        Train the Jordan RNN on the provided dataset.
        
        Args:
            X (Tensor): Input tensor of shape [sequence_length, input_size].
            y (Tensor): Target tensor of shape [sequence_length, output_size].
            epochs (int): Number of training epochs.
        
        Returns:
            Tuple: Final output tensor and the final loss value.
        """
        self.train()  # Set the model to training mode
        start_time = time.time()

        for epoch in range(epochs):

            total_loss = 0.0
            self.optimizer.zero_grad()  # Reset gradients at the start of each epoch
            output = self.init_output()  # Initialize output

            for i in range(X.shape[0]):
                input_i = X[i].to(self.device) # Shape: [input_size] or [batch_size, input_size]
                target_i = y[i].to(self.device)  # Shape: [output_size] or [batch_size, output_size]
                
                output = self.forward(input_i, output)  # Forward pass
                
                loss = self.loss_(output, target_i)  # Compute loss
                total_loss += loss

            # Backward pass and optimization
            total_loss.backward()
            self.optimizer.step()

            if (epoch + 1) % (epochs // 10) == 0 or epoch == 0:
                end_time = time.time()
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f} at time {end_time - start_time:.2f} seconds")
                start_time = time.time()


        return output, total_loss.item()
    

# add part where code is tested
if __name__ == "__main__":
    np.random.seed(42)

    # Parameters
    # n = 200  # Number of time points
    # time = np.arange(n)

    # # Simulating components of time series data
    # # 1. Trend component (linear trend)
    # trend = 0.1 * time

    # # 2. Seasonality component (sinusoidal pattern with periodicity)
    # seasonality = 10 * np.sin(2 * np.pi * time / 30)

    # # 3. Noise component (random noise)
    # noise = np.random.normal(0, 2, n)
    # time_series_data = trend + seasonality + noise

    # p = 1  # Number of features
    


    X = torch.tensor([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0]
])  # Shape: [sequence_length=4, input_size=2]

y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0]
])  # Shape: [sequence_length=4, output_size=1]

# Initialize the model
input_size = 2
hidden_size = 4
output_size = 1
activation = nn.ReLU()
loss_fn = nn.MSELoss()
optimizer_class = torch.optim.Adam
learning_rate = 0.01

model = JordanRNN(
    input_size=input_size, 
    hidden_size=hidden_size, 
    output_size=output_size, 
    activation=activation, 
    loss_eq=loss_fn, 
    optimizer=optimizer_class, 
    learning_rate=learning_rate,
    device=torch.device("mps")
)

# Verify the device of the model
# Train the model
epochs = 500
final_output, final_loss = model.fit(X, y, epochs=epochs)

print(f"Final Loss: {final_loss}")
print(f"Final Output: {final_output}")

