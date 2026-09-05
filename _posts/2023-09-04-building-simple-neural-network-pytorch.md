---
title: "Building a Simple Neural Network in PyTorch: Architecture, Training Loop, and Inference"
date: 2023-09-04
permalink: /posts/2023/09/building-simple-neural-network-pytorch/
blog_category: pytorch
blog_section: PyTorch Basics
blog_series: PyTorch Fundamentals
blog_order: 20
blog_summary: "Step through building, training, and evaluating a feedforward neural network in PyTorch using nn.Module, CrossEntropyLoss, SGD optimization, autograd backpropagation, and torch.no_grad() inference."
read_time: true
tags:
  - PyTorch
  - Neural Networks
  - Deep Learning
  - PyTorch Basics
---

Once you understand basic PyTorch tensors and operations, the next fundamental step in deep learning is building and training a **neural network**. PyTorch makes model creation modular and intuitive through `torch.nn.Module`, providing built-in layers, activation functions, loss functions, and optimization algorithms.

In this tutorial, we will build a complete end-to-end feedforward neural network in PyTorch:
1. Defining network architecture by subclassing `nn.Module`.
2. Understanding linear layers, weight shapes, and non-linear activations.
3. Executing a forward pass to inspect unnormalized outputs (**logits**).
4. Setting up a synthetic classification dataset and transferring data to GPU/CPU devices.
5. Configuring `nn.CrossEntropyLoss()` and `optim.SGD` optimizers.
6. Inspecting model layer weight and bias shapes.
7. Writing the canonical **PyTorch training loop** (zeroing gradients, backpropagation, and optimizer steps).
8. Evaluating the model using `model.eval()`, `torch.no_grad()`, and `torch.argmax()`.

---

## 1. Defining the Neural Network Architecture (`nn.Module`)

In PyTorch, all neural network architectures inherit from `torch.nn.Module`. Subclassing `nn.Module` gives your class built-in functionality for parameter tracking, device movement (`.to(device)`), state serialization (`state_dict()`), and GPU acceleration.

### Class Structure

A custom PyTorch model requires two primary methods:
- **`__init__(self, ...)`**: Defines and instantiates network layers as class attributes. Calling `super().__init__()` registers these layers so PyTorch automatically tracks their trainable parameters.
- **`forward(self, x)`**: Defines the computation executed on input tensor `x` during the forward pass. PyTorch uses this method to construct a dynamic computation graph on the fly.

### Mathematical Formulation of Layers

Our network consists of two fully connected (dense) linear layers:
1. **Fully Connected Layer 1 (`fc1`)**: Maps $10$ input features to $5$ hidden nodes.
2. **ReLU Activation**: Applies element-wise Rectified Linear Unit non-linearity: $f(z) = \max(0, z)$.
3. **Fully Connected Layer 2 (`fc2`)**: Maps $5$ hidden nodes to $2$ output logits for binary classification.

For a linear layer $y = x W^\top + b$:
- $x$ has shape $(\text{Batch}, \text{Input Size})$.
- $W$ has internal parameter shape $(\text{Output Size}, \text{Input Size})$.
- $b$ has bias vector shape $(\text{Output Size},)$.

```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        # First fully connected layer: input_size -> hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Second fully connected layer: hidden_size -> output_size
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Apply fc1 followed by ReLU activation
        x = torch.relu(self.fc1(x))
        # Pass through final linear layer (outputs raw logits)
        x = self.fc2(x)
        return x

# Instantiate model hyperparameters
input_size = 10
hidden_size = 5
output_size = 2

# Instantiate the network
model = SimpleNeuralNetwork(input_size, hidden_size, output_size)
print(model)
```

**Output:**

```text
SimpleNeuralNetwork(
  (fc1): Linear(in_features=10, out_features=5, bias=True)
  (fc2): Linear(in_features=5, out_features=2, bias=True)
)
```

---

## 2. Executing a Forward Pass & Understanding Logits

To test our model architecture, we pass a dummy input tensor containing $10$ samples (batch size of $10$), where each sample has $10$ features sampled from a standard normal distribution $\mathcal{N}(0, 1)$ via `torch.randn()`.

```python
# Create a random input tensor (batch_size=10, input_size=10)
input_tensor = torch.randn(10, input_size)

# Forward pass through the model
output_tensor = model(input_tensor)

print("Input Tensor (Shape: {}):".format(input_tensor.shape))
print(input_tensor)
print("\nOutput Tensor (Logits, Shape: {}):".format(output_tensor.shape))
print(output_tensor)
```

**Output:**

```text
Input Tensor (Shape: torch.Size([10, 10])):
tensor([[ 0.5981, -1.1226, -1.1484, -1.7993, -1.0522,  0.5901,  1.1900,  1.9547,  0.3601, -0.3148],
        [-1.1782,  0.8011, -0.7767, -0.8975,  0.2616, -0.4929, -0.0907,  1.1239, -0.9040, -0.6054],
        [-0.6976,  0.0796, -2.8259, -0.6161, -0.0825,  2.4683,  2.1527, -1.0783,  1.0604,  1.4096],
        [-0.4119, -0.4557, -1.4957,  1.9299,  0.4172,  0.5781, -0.8979, -0.9402,  0.4338, -0.4494],
        [-2.0288, -0.5040, -1.1490,  1.2484, -1.6605,  0.9228,  0.3663,  3.0035, -1.3805, -0.9023],
        [-0.1907, -0.3699,  0.5365, -0.4526,  0.8207,  1.0601,  2.3102, -0.1180,  0.4487, -0.6918],
        [ 2.4350,  0.8236, -0.5785,  0.0798,  1.1771,  2.7729,  0.4669,  1.0752,  2.0535,  1.5229],
        [-1.3552,  0.6298, -1.5841, -0.4133,  1.8091,  0.0422,  0.6711,  0.8808, -0.9581,  0.9971],
        [ 0.3043,  1.1972,  1.2524,  0.2678,  0.5470, -0.4037, -0.6721, -0.6063,  0.3090,  1.5853],
        [-0.6542, -0.9011, -0.6907, -0.6899, -0.4580, -3.2409,  0.3667,  0.1443,  0.3664,  0.8005]])

Output Tensor (Logits, Shape: torch.Size([10, 2])):
tensor([[0.1066, 0.3559],
        [0.1738, 0.5482],
        [0.1455, 0.2942],
        [0.1271, 0.1521],
        [0.4277, 0.4459],
        [0.1386, 0.3504],
        [0.2754, 0.5082],
        [0.2856, 0.4974],
        [0.2431, 0.4400],
        [0.0583, 0.1781]], grad_fn=<AddmmBackward0>)
```

Notice `grad_fn=<AddmmBackward0>` attached to the `output_tensor`. This indicates that PyTorch's **autograd engine** recorded the operations (`addmm` = matrix multiplication + bias addition) during the forward pass to enable gradient calculations during backpropagation.

---

## 3. Creating a Synthetic Training Dataset

Let's generate a synthetic training dataset containing $100$ samples:
- `x_train`: A matrix of shape $(100, 10)$ containing $100$ feature vectors.
- `y_train`: A vector of shape $(100,)$ containing random binary class targets ($0$ or $1$).

We also dynamically assign execution to a GPU (`cuda:0`) if available, or fallback to CPU.

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate 100 training samples with 10 features each
x_train = torch.randn(100, input_size).to(device)

# Generate 100 target class labels (integer 0 or 1)
y_train = torch.randint(0, output_size, (100,)).to(device)

print("Training Sample Inspection:")
print("x_train (first 2 samples):\n", x_train[:2])
print("y_train (first 2 targets):\n", y_train[:2])
```

**Output:**

```text
Training Sample Inspection:
x_train (first 2 samples):
tensor([[-0.7188, -1.5569, -0.1917, -0.1658,  0.8844,  1.1646,  0.1467, -1.8197, -0.7318,  0.4443],
        [ 0.3031, -0.2116, -1.3222, -1.3647, -0.8826,  1.3950, -0.8223,  1.0567, -0.7466,  0.1889]], device='cuda:0')
y_train (first 2 targets):
tensor([1, 1], device='cuda:0')
```

---

## 4. Defining Loss Function, Optimizer & Model Parameters

To train a neural network, we need two components:
1. **Loss Function (`nn.CrossEntropyLoss`)**: Measures how far our predicted logits are from the true class targets.
2. **Optimizer (`optim.SGD`)**: Updates trainable network parameters based on calculated gradients.

### Understanding `nn.CrossEntropyLoss()`

In PyTorch, `nn.CrossEntropyLoss()` combines `nn.LogSoftmax()` and `nn.NLLLoss()` (Negative Log Likelihood Loss) into a single numerically stable class.

$$\text{Loss}(x, y) = -\log \left( \frac{\exp(x_y)}{\sum_j \exp(x_j)} \right)$$

> **Crucial Rule:** Because `nn.CrossEntropyLoss()` computes Softmax internally, your network's final layer **must output unnormalized logits**, NOT manual `torch.softmax()` probabilities. Passing Softmax outputs into `CrossEntropyLoss` is a common bug that leads to degraded training dynamics.

### Optimizer & Layer Parameter Inspection

We instantiate Stochastic Gradient Descent (`optim.SGD`) with a learning rate $\eta = 0.01$.

```python
import torch.optim as optim

# Move model to selected computing device
model = SimpleNeuralNetwork(input_size, hidden_size, output_size).to(device)

# Loss function for multiclass/binary classification from raw logits
criterion = nn.CrossEntropyLoss()

# Optimizer: Stochastic Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Inspect model parameter shapes
print("Model Layer Parameter Shapes:")
print("fc1.weight.shape:", model.fc1.weight.shape)
print("fc1.bias.shape:  ", model.fc1.bias.shape)
print("fc2.weight.shape:", model.fc2.weight.shape)
print("fc2.bias.shape:  ", model.fc2.bias.shape)
```

**Output:**

```text
Model Layer Parameter Shapes:
fc1.weight.shape: torch.Size([5, 10])
fc1.bias.shape:   torch.Size([5])
fc2.weight.shape: torch.Size([2, 5])
fc2.bias.shape:   torch.Size([2])
```

- **`fc1.weight`**: $5 \times 10 = 50$ trainable weights + $5$ biases = $55$ parameters.
- **`fc2.weight`**: $2 \times 5 = 10$ trainable weights + $2$ biases = $12$ parameters.
- **Total Model Parameters**: $55 + 12 = 67$ learnable parameters.

---

## 5. The PyTorch Training Loop

The canonical PyTorch training loop executes four core operations during every epoch:

1. **Forward Pass**: Compute predicted outputs `outputs = model(x_train)`.
2. **Loss Calculation**: Evaluate loss `loss = criterion(outputs, y_train)`.
3. **Zero Gradients (`optimizer.zero_grad()`)**: Clear gradient buffers from the previous step. By default, PyTorch **accumulates** gradients on calls to `.backward()` rather than overwriting them.
4. **Backward Pass (`loss.backward()`)**: Traverse the computational graph to compute partial derivatives $\frac{\partial \mathcal{L}}{\partial w}$ for all parameters with `requires_grad=True`.
5. **Optimizer Step (`optimizer.step()`)**: Update parameter weights according to the optimizer update rule: $w \leftarrow w - \eta \cdot \nabla_w \mathcal{L}$.

```python
num_epochs = 100

print("Starting Training Loop...")
for epoch in range(num_epochs):
    # 1. Forward pass
    outputs = model(x_train)

    # 2. Compute loss
    loss = criterion(outputs, y_train)

    # 3. Backward pass and optimization
    optimizer.zero_grad()  # Reset parameter gradient buffers
    loss.backward()        # Calculate gradients via backpropagation
    optimizer.step()       # Update model parameters

    # Print loss progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
```

**Output:**

```text
Starting Training Loop...
Epoch [10/100], Loss: 0.7424
Epoch [20/100], Loss: 0.7344
Epoch [30/100], Loss: 0.7274
Epoch [40/100], Loss: 0.7214
Epoch [50/100], Loss: 0.7163
Epoch [60/100], Loss: 0.7117
Epoch [70/100], Loss: 0.7077
Epoch [80/100], Loss: 0.7043
Epoch [90/100], Loss: 0.7013
Epoch [100/100], Loss: 0.6986
```

As optimization progresses across $100$ epochs, gradient descent successfully updates weights to minimize the cross-entropy loss from $0.7424$ down to $0.6986$.

---

## 6. Model Evaluation and Inference

Once training finishes, we evaluate the trained model on unseen test inputs.

### Best Practices for Evaluation

1. **`model.eval()`**: Sets the model to evaluation mode. While our simple network lacks layers like `nn.Dropout` or `nn.BatchNorm2d`, calling `model.eval()` is essential practice because it disables dropout randomness and switches batch normalization to use running population statistics.
2. **`with torch.no_grad():`**: A context manager that disables autograd history tracking during inference. This reduces memory usage and speeds up matrix operations since PyTorch does not need to store intermediate activation tensors for backward gradient computations.
3. **`torch.argmax(logits, dim=1)`**: Converts raw output logits into predicted discrete class indices ($0$ or $1$) along dimension $1$ (across output class columns).

```python
# Set model to evaluation mode
model.eval()

# Disable autograd gradient computation for inference
with torch.no_grad():
    # Create 5 test samples
    test_input = torch.randn(5, input_size).to(device)
    test_output = model(test_input)

    print("Test Input Shape:", test_input.shape)
    print("\nRaw Output Logits (Shape: {}):".format(test_output.shape))
    print(test_output)

    # Predict discrete class labels using argmax
    predicted_classes = torch.argmax(test_output, dim=1)
    print("\nPredicted Class Labels (0 or 1):")
    print(predicted_classes)
```

**Output:**

```text
Test Input Shape: torch.Size([5, 10])

Raw Output Logits (Shape: torch.Size([5, 2])):
tensor([[ 0.4773, -0.0853],
        [-0.1125, -0.3616],
        [ 0.2163, -0.4997],
        [ 0.2524, -0.0185],
        [-0.0031, -0.2400]], device='cuda:0')

Predicted Class Labels (0 or 1):
tensor([0, 0, 0, 0, 0], device='cuda:0')
```

In the test predictions above:
- For sample 1: Logit 0 ($0.4773$) > Logit 1 ($-0.0853$) $\to$ Predicted Class = `0`.
- For sample 2: Logit 0 ($-0.1125$) > Logit 1 ($-0.3616$) $\to$ Predicted Class = `0`.

---

## Summary

In this tutorial, we implemented a complete feedforward neural network pipeline in PyTorch:
1. **Model Definition**: Subclassing `nn.Module`, initializing `nn.Linear` layers, and specifying the forward execution graph with `torch.relu()`.
2. **Logits & Loss**: Understanding that `nn.CrossEntropyLoss` expects raw unnormalized logits from linear layers.
3. **Training Loop Pattern**:
   - `outputs = model(inputs)` (Forward pass)
   - `loss = criterion(outputs, targets)` (Loss computation)
   - `optimizer.zero_grad()` (Reset accumulated gradients)
   - `loss.backward()` (Backpropagate gradients)
   - `optimizer.step()` (Update parameter weights)
4. **Inference**: Disabling gradient computation with `with torch.no_grad():` and extracting target class predictions using `torch.argmax(logits, dim=1)`.

With these PyTorch basics in place, you can build deeper neural network architectures, train on real-world datasets with `DataLoader`, and implement advanced optimization algorithms!
