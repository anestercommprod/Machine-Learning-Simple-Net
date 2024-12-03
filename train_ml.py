# PyTorch ML implementation

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn

writer = SummaryWriter()

num_samples = 1000
inputs = np.random.randint(0, 11, num_samples)  # Random integers from 0 to 10
targets = inputs.astype(float)  # Output should be the same as input



class CopyNumberModel(nn.Module):
    def __init__(self):
        super(CopyNumberModel, self).__init__()
        self.fc = nn.Linear(1, 1)  # Converitng single input to single output

    def forward(self, x):
        return self.fc(x)

model = CopyNumberModel()
criterion = nn.MSELoss()  # MSE loss func
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

inputs_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1)
targets_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)



epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs_tensor)
    loss = criterion(outputs, targets_tensor)
    loss.backward()
    optimizer.step()
    writer.add_scalar("Loss/train", loss.item(), epoch) # write tensorboard
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "ml_based_model.pth")
