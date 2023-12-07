import torch
import torch.nn as nn
import torch.optim as optim
from RNN import RNN_ManyToMany


def train_model(model, train_loader, criterion, optimizer, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Reshape outputs and targets if needed
            # outputs = outputs.view(targets.size())  # Example: reshape if needed

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {epoch_loss:.4f}")

    print('Training finished')

# Example usage:

# Define your dataset and data loader
# train_dataset = YourDataset(...)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize your model
input_size = 10  # Replace with your input size
hidden_size = 20  # Replace with your hidden size
model = RNN_ManyToMany(input_size, hidden_size)  # Replace with your desired model

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
# train_model(model, train_loader, criterion, optimizer, epochs)
