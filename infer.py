import torch
import torch.nn as nn

class CopyNumberModel(nn.Module):
    def __init__(self):
        super(CopyNumberModel, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

model = CopyNumberModel()
model.load_state_dict(torch.load("reward_based_model.pth")) # To use ML-based model replace with `ml_based_model.pth` once  model is trained and saved
model.eval() 

print("Model is loaded. Enter int/float numbers to test it.")

while True:
    try:
        user_input = input("Input a number: ")
        if user_input.lower() == "exit":
            break
        input_number = float(user_input)
        
        input_tensor = torch.tensor([[input_number]], dtype=torch.float32)
        
        with torch.no_grad():
            output = model(input_tensor)
        print(f"Model output: {output.item():.4f}")
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except ValueError:
        print("NaN - Enter a valid number.")
