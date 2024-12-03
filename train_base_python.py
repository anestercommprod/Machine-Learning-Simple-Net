# Framework-less implementation of the same neural-net

import random

weights = [random.uniform(-1, 1)]
bias = random.uniform(-1, 1)

learning_rate = 0.01
epochs = 25000

reward_multiplier = 1.0

def forward(x):
    return weights[0] * x + bias

def compute_loss(target, prediction):
    error_margin = abs(target - prediction)
    non_integer_penalty = abs(prediction - round(prediction))
    reward = reward_multiplier * (1.0 - error_margin) - non_integer_penalty
    return -reward  # Loss is a negative reward

for epoch in range(epochs):
    input_value = random.uniform(0, 10)
    target = input_value

    output = forward(input_value)

    loss = compute_loss(target, output)

    gradient_w = -2 * (target - output) * input_value  # Derivative wrt weight
    gradient_b = -2 * (target - output)  # Derivative wrt bias

    weights[0] -= learning_rate * gradient_w
    bias -= learning_rate * gradient_b

    # Print progress once per 1000 epochs
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Weight: {weights[0]:.4f}, Bias: {bias:.4f}")



test_inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 10, 123, 123123123123]
print("\nTesting the model at epoch " + str(epochs) + ":")
for test_input in test_inputs:
    prediction = forward(test_input)
    print(f"Input: {test_input}, Output: {prediction:.4f}")