# PyTorch DL implementation utilizing RL with extended logs

import torch
import torch.nn as nn
import torch.optim as optim
import random

class RewardBasedModel(nn.Module):
    def __init__(self):
        super(RewardBasedModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

model = RewardBasedModel()
optimizer = optim.SGD(model.parameters(), lr=0.01) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 25000
reward_multiplier = 1.0  # Scale rewards/penalties

log_file = "reward_based_training.log"
with open(log_file, "w") as f:
    f.write("Epoch,Input,Output,Error Margin,Reward\n")

for epoch in range(epochs):
    input_number = random.uniform(0, 10)  # Random inputs
    input_tensor = torch.tensor([[input_number]], dtype=torch.float32).to(device)

    output = model(input_tensor)
    
    target = input_tensor.clone().detach()  # Target is the input itself
    error_margin = torch.abs(output - target)
    reward = reward_multiplier * torch.exp(-error_margin)

    penalty = torch.where(error_margin > 1.0, error_margin, torch.zeros_like(error_margin))
    reward -= penalty

    loss = -reward.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with open(log_file, "a") as f:
        f.write(f"{epoch+1},{input_number:.4f},{output.item():.4f},{error_margin.item():.4f},{reward.item():.4f}\n")

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}/{epochs}: Input={input_number:.4f}, Output={output.item():.4f}, Error Margin={error_margin.item():.4f}, Reward={reward.item():.4f}")

model_file_name = "reward_based_model.pth"
torch.save(model.state_dict(), model_file_name)
print(f"Training has been completed. Model saved as '{model_file_name}'. Logs saved to '{log_file}'.")