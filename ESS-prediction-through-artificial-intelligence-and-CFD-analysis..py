import torch
import torch.nn as nn
import torch.optim as optim

# Define input data and target data
inputs = torch.tensor([
    [1, 65, 90, 0],
    [1, 65, 90, 70],
    [1, 65, 90, 80],
    [1, 65, 90, 90],
    [1, 70, 90, 0],
    [1, 70, 90, 70],
    [1, 70, 90, 80],
    [1, 70, 90, 90],
    [1, 75, 90, 0],
    [1, 75, 90, 70],
    [1, 75, 90, 80],
    [1, 75, 90, 90],
    [0, 0, 90, 0],
    [0, 0, 90, 70],
    [0, 0, 90, 80],
    [0, 0, 90, 90]
], dtype=torch.float32)

targets = torch.tensor([
    [62.69732, 0.2461883],
    [138.28, 0.2003664],
    [280.5077, 0.2329488],
    [888.225, 0.2573726],
    [71.15601, 0.3098813],
    [138.5782, 0.2593856],
    [301.726, 0.4405224],
    [894.8782, 0.2968768],
    [35.71762, 0.5227968],
    [140.0325, 0.2833194],
    [282.8187, 0.2549224],
    [894.7023, 0.5579914],
    [22.41116, 3.517495],
    [99.36477, 0.1736754],
    [203.5698, 0.3146114], 
    [661.348, 1.079459]
], dtype=torch.float32)

# MLP model definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# model initialization
model = MLP()

# Loss function and optimizer definition
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# model training
num_epochs = 7000
for epoch in range(num_epochs):
    # Forward propagation
    outputs = model(inputs)
    
    # loss calculation
    loss = criterion(outputs, targets)
    
    # Backpropagation and weight update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Prediction on new inputs
new_input = torch.tensor([[0, 0, 90, 0]], dtype=torch.float32)
with torch.no_grad():
    prediction = model(new_input)

print("Prediction:", prediction)