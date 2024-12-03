# Machine-Learning-Simple-Net

A very simple PyTorch / pure model that learns to output matching numbers.
This is a single-layer neural net model with linear activation made by utilizing RL and MSE approaches.

Running this project will require a PC that has unit to process FP32, is compatible with PyTorch, at least 4 GB of ram and ~5 GB of free disk space.

---
# Machine-Learning approach deep-dive (`train_ml.py`)

At first, I decided to train the model utilizing standard Machine-Learning techniques, with predefined inputs and expected outputs. This approach uses Mean Squared Error (MSE) as the loss function and Stochastic Gradient Descent(SGD) as the optimizer.

### Model Definition

The `CopyNumberModel` class is a simple neural network with one input neuron and one output neuron. The model does not use any activation function, making it suitable for regression tasks where the output should be continuous.

### Data Preparation

The script generates random integer inputs between 0 and 10 using `np.random.randint`. These inputs are converted to PyTorch tensors and reshaped to have a single feature dimension(`unsqueeze(1)`).

### Loss Function and Optimizer

MSE is used as the loss function, which is appropriate for regression tasks. Stochastic Gradient Descent(SGD) with a learning rate of 0.01 is chosen as the optimizer.

### Training Loop

The script runs for 500 epochs. In each epoch, it performs the following steps:
- Zeroes out the gradients of the model parameters using `optimizer.zero_grad()`.
- Passes inputs through the model to get predictions(`outputs`).
- Calculates the loss between model predictions and target input.
- Back-propagates the loss to compute gradients(`loss.backward()`).
- Updates the model parameters using the optimizer(`optimizer.step()`).

This script demonstrates a simple machine learning implementation using PyTorch, including data preparation, model definition, loss function, optimizer, and training loop. The architecture is very basic and consists of a linear transformation without any activation function, making it suitable for regression tasks where the output should be continuous.

### Pre/Post Training

To get somewhat usable results from the model, we should train it for at least 500 epochs which takes about 0.001 second per epoch on a consumer-grade RTX-based x86 computer.

Let's explore model performance trained for various epochs(10, 50, 250, 500).
To simply my testings, I will only include three numbers as my input: `1`, `123` and `123123123123` to show each model performance.

#### Epoch 10
```
Input a number: 1
Model output: 1.3649
Input a number: 123
Model output: 115.7177
Input a number: 123123123123
Model output: 115405479936.0000
```

#### Epoch 50
```
Input a number: 1
Model output: 0.7919
Input a number: 123
Model output: 127.1819
Input a number: 123123123123
Model output: 127553527808.0000
```

#### Epoch 250
```
Input a number: 1
Model output: 0.8742
Input a number: 123
Model output: 125.4488
Input a number: 123123123123
Model output: 125721346048.0000
```

#### Epoch 500
```
Input a number: 1
Model output: 1.0394
Input a number: 123
Model output: 122.2405
Input a number: 123123123123
Model output: 122316914688.0000
```

Let's take a look at model loss during training.

```
Epoch [10/10], Loss: 0.1725
...
Epoch [50/50], Loss: 0.1119
...
Epoch [250/250], Loss: 0.0061
...
Epoch [500/500], Loss: 0.0000
```

As the number of epochs increases, the model's predictions become closer to the actual input values, `loss` during training decreases significantly as the number of epochs increases, showing that the model is learning effectively, though not as effective as it could have had - model still struggles to properly output number bigger than two digits.

Model accuracy decreases as number gets higher because MSE is a suitable loss function for regressive tasks, it doesn't penalise larger errors as heavily as smaller ones, which makes it *harder* for the model to learn from larger inputs that are not in the input list(we only specified inputs from 0 to 10).

Though, we have trained the model with Learning Rate(LR) of `0.01`, how will accuracy change if we will drop LR to `0.001`? 

```
Epoch [500/500], Loss: 0.0164
```

Our *loss* become higher, because lower LR requires more epochs to go through for model to properly converge to target value.
```
Input a number: 1
Model output: 0.7929
Input a number: 123
Model output: 127.0430
Input a number: 123123123123
Model output: 127412420608.0000
```

Let's put this model through 5000 epochs using `Learning Rate` of `0.001`:
```
Epoch [5000/5000], Loss: 0.0003
```
```
Input a number: 1
Model output: 1.0278
Input a number: 123
Model output: 122.4605
Input a number: 123123123123
Model output: 122550575104.0000
```

Apparently, model that's been through 5000 iterations is not much better than model with lower `LR` that's only been through 500 iterations, and this raises very important topic:

### Why Lowering Learning Rate Improves Convergence?

Lower `LR` makes model to take smaller steps during each weights update, this might lead to more stable convergence as model has a better chance of finding a minimum that is closer to the global minimum. But why this didn't work out in this case?

Any loss function will essentially hit its plateau - a certain number of epochs after which model stops learning anything new. Training model for more epochs requires more computational resources and time and might not be necessary as we can always change how exactly our model learns and tickle with it.
Another downside of putting model through too many epochs is overfitting: with very small steps(smaller LR) model might start to overfit to the training data, especially if dataset is either very limited or low quality. This is very important to keep the balance between loss, LR and the amount of epochs.

Though, there's a space for improvement - let's discuss how I happen to improve model performance utilizing RL technique.

# Deep-Learning approach (`train_reward.py`)

After seeing how Machine-Learning performs, I decided to try to replicate this simple single-layer, one input neuron model behaviour with Deep Learning and RL(Reinforcement learning) technique.

### What's RL?

Reinforcement Learning(RL) is a type of Machine Learning where model learns to interact with given environment by performing actions and receiving rewards of penalties. The goal of the model is to maximize total sum of rewards and minimize penalties over time, this helps model to learn optimal policies for making decisions.
OpenAI has published their research paper in 2019 that explains this topic further - [Emergent tool use from multi-agent interaction](https://openai.com/index/emergent-tool-use/).

### Model Definition

As our model environment is a simple regression task where the goal is to predict the same number as input, we can apply RL to train the model. We will define a reward function that gives high rewards when the model's output matches the target and penalizes large errors.

The `RewardBasedModel` class is a simple neural network with one input neuron and one output neuron. The model does not use any activation function, making it suitable for regression tasks where the output should be continuous - all remains the same as in Machine-Learning approach.

### Data Preparation

The script generates random integer inputs between 0 and 10 using `np.random.randint`. These inputs are converted to PyTorch tensors and reshaped to have a single feature dimension(`unsqueeze(1)`).

### Loss Function and Optimizer

The reward-based loss function is derived from the reward, which is a custom scalar value that guides the model's learning process. The optimizer used here is the same Stochastic Gradient Descent(SGD) with a learning rate of 0.01.

### Model Training

Since this is a Deep-Learning approach, this method requires much more epochs to go through as model *learns* what's good and what's bad for its task - we haven't predefined any rules for it to learn from.

Let's try to train the model and play with it.
A very important note: initially, I used different `reward` than you can see in the script on this github page:
`reward = reward_multiplier * (1.0 - error_margin)`, therefore I will demonstrate that version of the script first.

### Post/Pre Training

If training time for my ML approach took less than a second(aside from loading up PyTorch components), Deep Learning takes much more time, as well as much more computational resources to achieve the same result.

To measure your training time on your system you can copy `start_timestamp` and `end_timestamp` from `train_ml.py` script.
```
Epoch 1000/25000: Input=8.6513, Output=8.3372, Error Margin=0.3141, Reward=0.6859
Epoch 10000/25000: Input=7.9881, Output=7.3819, Error Margin=0.6063, Reward=0.3937
Epoch 25000/25000: Input=6.0570, Output=5.9156, Error Margin=0.1414, Reward=0.8586
Training took 27.03475522994995 seconds.
```

Training this deep-learning model for 25000 epochs took me 27 seconds - quite a lot for such a simple task.
During training process GPU was utilized for over 90% and the training process filled 224 megabytes of video memory.

#### Why Deep-Learning takes so much resources?

Deep Learning models, especially the ones with more than a few layers, have a much higher number of parameters.
For example, my `RewardBasedModel` has only one parameter, but models that are designed for more complex tasks can have tens of thousands, millions, or even billions of active parameters. Such models also require much higher batch_size.

In my case, we have set batch_size to 1 at this line:
```
input_tensor = torch.tensor([[input_number]], 
```
Each iteration of the training loop processes one sample(`input_tensor`), computes the output and updates the model parameters. This single-sample approach is typical for simple models like the one we are training in this article.
As of why I set batch_size to 1: Simplicity. For a very simple model with just one parameter processing one sample at a time is straightforward and easy to understand, not even speaking of resource efficiency as higher batch_size will require much more RAM/VRAM.

For this specific script, using a batch size of 1 results in a *relatively* short training time

#### Testing Deep-Learning Model

```
Input a number: 1
Model output: 1.0576
Input a number: 123
Model output: 127.3472
Input a number: 123123123123
Model output: 127452110848.0000
```

We can clearly see that DL model is not even outperforming ML model. 
*Why? Do we need to put the model through more epochs?*
-Let's try!

```
Epoch 1000/100000: Input=4.3159, Output=4.2352, Error Margin=0.0807, Reward=0.9193
Epoch 10000/100000: Input=1.3050, Output=1.3157, Error Margin=0.0107, Reward=0.9893
Epoch 25000/100000: Input=6.1854, Output=6.7163, Error Margin=0.5309, Reward=0.4691
Epoch 50000/100000: Input=1.7424, Output=1.7578, Error Margin=0.0154, Reward=0.9846
Epoch 100000/100000: Input=4.9999, Output=5.3344, Error Margin=0.3345, Reward=0.6655
Training took 109.99232578277588 seconds.
```

```
Input a number: 1
Model output: 1.0014
Input a number: 123
Model output: 125.2339
Input a number: 123123123123
Model output: 125376249856.0000
```

\- This is still ain't no better than Machine-Learning model, regardless how many epochs it is going through.
Do you still remember about `reward` variable? This is the key point in RL Deep-Learning to properly adjust the model to its actual goal.

Let's set epochs amount back to 25000, replace `reward = reward_multiplier * (1.0 - error_margin)` with what is actually in my github script and set `reward_multiplier` to `0.50`:
```
reward = reward_multiplier * torch.exp(-error_margin)
```

Now we can retrain the model and see how it's accuracy improved with much more strict reward/penalty ratio.

```
Epoch 1000/50000: Input=9.1879, Output=9.5567, Error Margin=0.3688, Reward=0.6915
Epoch 10000/50000: Input=7.2859, Output=7.7552, Error Margin=0.4692, Reward=0.6255
Epoch 25000/50000: Input=3.2489, Output=3.3827, Error Margin=0.1338, Reward=0.8748
Epoch 50000/50000: Input=0.5709, Output=0.5805, Error Margin=0.0097, Reward=0.9904
Training took 27.180826902389526 seconds.
```

Model took the exact same time to train as initial version with 25000 epochs, let's see the results.

```
Input a number: 1
Model output: 0.9998
Input a number: 123
Model output: 123.7488
Input a number: 123123123123
Model output: 123879079936.0000
```

This is much, much better than all the previous attempts to create AI that will replicate input numbers.
Let's devide `-error_margin` in the reward variable by `10` and see how this will affect model's behaviour:

```
Epoch 1000/25000: Input=1.4402, Output=1.4441, Error Margin=0.0039, Reward=0.9996
Epoch 10000/25000: Input=3.6539, Output=3.6513, Error Margin=0.0026, Reward=0.9997
Epoch 25000/25000: Input=3.5661, Output=3.5526, Error Margin=0.0136, Reward=0.9986
```

```
Input a number: 1
Model output: 1.0015
Input a number: 123
Model output: 122.9376
Input a number: 123123123123
Model output: 123058642944.0000
```

We can clearly see that model has improved its accuracy, yet it still not perfect. -*Why?*
Now this is clear that the issue is in how our reward/penalty system is constructed. This is why I spent so much time sharing my observations about Deep-Learning model's performance using various parameters for reward system.

Now, we need to actually configure reward/penalty system in a way that will guide the model in the right direction.
```
epochs = 25000
reward_multiplier = 1.0
...
    reward = reward_multiplier * torch.exp(-error_margin / 100)

    penalty = torch.where(error_margin > 0.01, error_margin * 1.333, torch.zeros_like(error_margin))
```

Here I configured `reward_multiplier` to my initial default of `1.0`, devided `-error_margin` by `100` in `reward` variable and applied some changes to penalty: `error_margin > 0.01` and multiplied `error_margin` by the factor of `1.333`.

Results during model training:
```
Epoch 1000/25000: Input=7.5851, Output=7.5821, Error Margin=0.0029, Reward=1.0000
Epoch 10000/25000: Input=3.5506, Output=3.5493, Error Margin=0.0013, Reward=1.0000
Epoch 25000/25000: Input=6.5073, Output=6.5114, Error Margin=0.0040, Reward=1.0000
```

Results in manual testing:
```
Input a number: 1
Model output: 0.9999
Input a number: 9
Model output: 8.9949
Input a number: 123
Model output: 122.9963
Input a number: 123123123123
Model output: 123119566848.0000
```

This model is very close to almost perfect - though, still not yet.
Let me try to break down why PyTorch will result in this kind of behaviour.

- PyTorch utilizes weights randomly(Xavier/Kaiming initialization), this means that model might start farther from the ideal solution that we will discuss later on.
- PyTorch uses generic optimisers(just like the one I used - SGD), they work on a wide range of tasks, but are not perfect for my task - *input = output*.
- PyTorch LR schedulers and gradients are made for stability, scalability and generalisation - this might make convergence of the model slower for this specific task. Also gradient updates are adding small noise, requiring more epochs to smooth these errors out. 
- PyTorch uses FP32 blocks of the GPU; this introduces tiny inaccuracies like rounding errors, especially when dealing with a very small or very large numbers. To smooth this out we might need to train for more epochs.

It is possible to train such a simple model with one active input neuron on PyTorch that will be as good as a basic mathematical model we'll soon discuss, but this requires more time to discover perfect setup for such a model in PyTorch, as well as setting weights and biases to values closer to the expected solution and making loss/reward more aggressive.

My final point: PyTorch models are designed for versatility and generalisation, which adds complexity but enables broader applications, which is why we now have models like RasNet, BERT, GPTs, StableDiffusion and more.

# Pure Mathematical Model (`train_base_python.py`)

This implementation is framework-free and can be done in any language as it requires basic understanding of how such neural nets are made.
This script represents the same single-layer, linear-regressive model, and instead of one *active neuron* I am using a single linear transformation to map out the input value to an output value. Let me share more information on what active neuron is.

*Active* can be interpreted as a neuron that is involved in the computation, and since this script uses a single linear layer (`nn.Linear(1,1)`) - we only have one input neuron, the single feature in the input tensor.



This is crucial to differenciate that this mathematical model is not a traditional neural network with layers and non-linear activations, it is more of a simple mathematical transformation where the input directly influences the output through a linear function.

This being mathematical model, we still have a few features of a neural net:
- `RewardBasedModel` with a single linear layer that takes one input and produces one output, and parameters of this layer are *learnable*.
- `forward` function that manually computes the output using a linear transformation(`output = weights[0] * x + bias`).
- Training loop that iterates over random inputs and updates the model parameters manually using gradient descent.

Mathematical models are very powerful tool when it comes to something **simple** - something that has no complexity, this is where they shine and where traditional neural nets are not as efficient due to their nature. 
Traditional Neural Networks offer high flexibility in terms of architecture and can be adapted to various types of problems. The mathematical model is limited to a specific type of transformation: input to output.

### Testing the model

Let's test how it compares to our traditional neural net variant:
```
Epoch 1000/25000, Loss: -0.5020, Weight: 1.0002, Bias: -0.0012
Epoch 10000/25000, Loss: -0.9396, Weight: 1.0000, Bias: -0.0000
Epoch 25000/25000, Loss: -0.5078, Weight: 1.0000, Bias: -0.0000
Training took 0.018016815185546875 seconds.
```

Already looks promising. But how well this actually performs?

```
Testing the model at epoch 25000:
Input: 1, Output: 1.0000
Input: 2, Output: 2.0000
Input: 3, Output: 3.0000
Input: 4, Output: 4.0000
Input: 5, Output: 5.0000
Input: 6, Output: 6.0000
Input: 7, Output: 7.0000
Input: 8, Output: 8.0000
Input: 9, Output: 9.0000
Input: 0, Output: -0.0000
Input: 10, Output: 10.0000
Input: 123, Output: 123.0000
Input: 123123123123, Output: 123123123123.0000
```

\- This is exactly what I've been trying to achieve with PyTorch - a model that can match its output with input.
Even if we *train* this model for 2500 epochs, it's already better than how traditional Neural Nets can perform:

```
Testing the model at epoch 2500:
Input: 1, Output: 1.0000
Input: 2, Output: 2.0000
Input: 3, Output: 3.0000
Input: 4, Output: 4.0000
Input: 5, Output: 5.0000
Input: 6, Output: 6.0000
Input: 7, Output: 7.0000
Input: 8, Output: 8.0000
Input: 9, Output: 9.0000
Input: 0, Output: 0.0000
Input: 10, Output: 10.0000
Input: 123, Output: 122.9999
Input: 123123123123, Output: 123123019917.7965
```

# Conclusion

This research explored three different approaches to create a simple neural net model that learns to output matching numbers: Machine-Learning, Deep Learning with RL and Pure Mathematical Model. Each approach has its strenghts and limitations, which we discussed in details earlier.

### Machine-Learning Implementation

ML approach utilized predefined inputs and expected outputs, employing `MSE` as the loss function and `SGD` gradient as optimiser.
Models that are trained in all the examples is a single-layer model with one input and output *neuron*. While Machine-Learning approach performed well for small inputs that were in the dataset, it struggled with larger numbers due to the nature of MSE, which does not penalize large errors heavily.

### Deep-Learning Implementation

In this implementation I introduced Reinforcement Learning to the model training - this makes a reward-based loss function that guides the model during the learning process based on error margins. Model was trained for 25000 epochs, demonstrating significant improvements in accuracy in comparison to the original Machine-Learning approach.

### Pure Mathematical Implementation

This model used a single linear transformation to map inputs and outputs without any activation function - this implementation demonstrated excellent accuracy, matching input values exactly and was the cheapest to compute and run. Simplicity of this model makes it particularly efficient for tasks with low complexity.

### Comparison and Insights

Comparing the three approaches, we found that:

- **ML Implementation**: Effective for small inputs but struggles with larger numbers without RL.
- **RL Implementation**: Drastically improves accuracy with properly configured reward functions.
- **Pure Mathematical Implementation**: Excellent performance, more efficient for simple tasks.

The choice of model implementation heavily depends on the specific requirement of the task. For simple regression tasks, mathematical models might fit best due to their efficiency and accuracy in low-complexity tasks, and on the other hand for more complex tasks that require learning from data, `ML`/`DL` approach with `RL` will often be a better option to go with.

### Undisclosed

I haven't spoke much about hyperparameter tuning, model architecture and performance metrics other than loss and margin error because of the simplicity of the model task, as this is not exactly neccessary for this particular case - this is a topic that I will uncover in models that require at least a few active and hidden layers to solve the task.

To summarise this research further: You, of course, can try to recreate neural network from scratch to fit your very own usecase, but it will take anormous amount of time if given task is far too complex. What matters more is how one guides AI to learn - what techniques were used, how training environment was configured, what dataset is used and what architecture/optimisers/gradients are used.

---

I would appreciate any feedback on this *research paper*, preferably as *\*Reinforcement Learning with Human Feedback\**.
