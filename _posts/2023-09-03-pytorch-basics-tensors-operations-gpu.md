---
title: "PyTorch Basics: Tensors, Operations, and GPU Acceleration"
date: 2023-09-03
permalink: /posts/2023/09/pytorch-basics-tensors-operations-gpu/
redirect_from:
  - /posts/2026/09/pytorch-basics-tensors-operations-gpu/
  - /posts/2026/09/pytorch-basics-tensors-operations-gpu
blog_category: pytorch
blog_section: PyTorch Basics
blog_series: PyTorch Fundamentals
blog_order: 10
blog_summary: "A practical guide to PyTorch fundamentals: installing PyTorch, understanding multidimensional tensors, mastering indexing and reshaping, performing tensor operations, and measuring GPU acceleration."
read_time: true
tags:
  - PyTorch
  - Tensors
  - GPU Acceleration
  - Deep Learning
  - PyTorch Basics
---

PyTorch is an open-source deep learning framework developed by Meta AI (FAIR) to simplify building neural networks and machine learning models. Built around dynamic computational graphs (eager execution) and Pythonic ergonomics, PyTorch combines intuitive tensor manipulation with hardware acceleration on GPUs and specialized AI chips.

In this tutorial, we will cover core PyTorch concepts from scratch:
1. Setting up a Python virtual environment and installing PyTorch.
2. Understanding $N$-dimensional **tensors** and tensor creation routines.
3. Manipulating tensors through **indexing, slicing, and reshaping**.
4. Performing **tensor operations, concatenation, and element-wise arithmetic**.
5. Comparing execution performance between **CPU vs. GPU**, with GPU memory profiling.

---

## 1. Environment Setup & Installation

Before writing PyTorch code, it is best practice to create an isolated Python virtual environment. This keeps project dependencies clean and avoids version conflicts with system packages.

### Creating a Virtual Environment

```bash
# Create a virtual environment named 'myvenv'
python3 -m venv myvenv

# Activate the virtual environment
# On Linux / macOS:
source myvenv/bin/activate

# On Windows Command Prompt / PowerShell:
# myvenv\Scripts\activate
```

### Installing PyTorch

PyTorch provides three core packages:
- `torch`: The core tensor computation library and dynamic autograd engine.
- `torchvision`: Datasets, model architectures, and image transformations for computer vision.
- `torchaudio`: Data loaders, audio processing functions, and pretrained models for audio processing.

```bash
pip install torch torchvision torchaudio
```

> **Note:** If you have an NVIDIA GPU with CUDA support, visit the official [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) to obtain the exact `pip` or `conda` command matching your CUDA driver version (e.g., CUDA 11.8 or CUDA 12.1).

---

## 2. What is a Tensor?

A **tensor** is a multidimensional array of homogeneous data types (e.g., 32-bit floats or 64-bit integers). Tensors are the fundamental data structures in PyTorch, serving the same role as NumPy arrays while providing two key features necessary for deep learning:
1. **GPU/TPU Acceleration**: Tensors can be transferred to hardware accelerators for parallel computation.
2. **Automatic Differentiation**: PyTorch tracks operations on tensors to automatically compute gradients (`autograd`).

### Tensor Dimensionality Hierarchy

- **$0$-D Tensor**: A single scalar value (e.g., `torch.tensor(5)`).
- **$1$-D Tensor**: A vector of values with shape `(N,)`.
- **$2$-D Tensor**: A matrix with shape `(Rows, Columns)`.
- **$3$-D Tensor**: A $3$-dimensional array with shape `(Channels/Batch, Height, Width)` or similar sequences.

### Creating Tensors in PyTorch

PyTorch provides multiple tensor initializers: explicit values, random distributions, and fixed values like zeros or ones.

```python
import warnings
warnings.filterwarnings("ignore")  # Clean notebook output

import torch

# 1-D Tensor (Vector)
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
print("1-D Tensor:")
print(tensor_1d)

# 2-D Tensor (Matrix)
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("\n2-D Tensor:")
print(tensor_2d)

# 3-D Tensor
tensor_3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("\n3-D Tensor:")
print(tensor_3d)

# Random Tensor (3x4 float tensor sampled from Uniform(0, 1))
random_tensor = torch.rand(3, 4)
print("\nRandom 3x4 Tensor:")
print(random_tensor)

# Zero Tensor (2x3 tensor filled with 0.0)
zero_tensor = torch.zeros(2, 3)
print("\nZero Tensor (2x3):")
print(zero_tensor)

# One Tensor (2x3 tensor filled with 1.0)
one_tensor = torch.ones(2, 3)
print("\nOne Tensor (2x3):")
print(one_tensor)
```

**Output:**

```text
1-D Tensor:
tensor([1, 2, 3, 4, 5])

2-D Tensor:
tensor([[1, 2, 3],
        [4, 5, 6]])

3-D Tensor:
tensor([[[1, 2],
         [3, 4]],

        [[5, 6],
         [7, 8]]])

Random 3x4 Tensor:
tensor([[0.7003, 0.6034, 0.2690, 0.1023],
        [0.4210, 0.9153, 0.9505, 0.0278],
        [0.8169, 0.2516, 0.8302, 0.5712]])

Zero Tensor (2x3):
tensor([[0., 0., 0.],
        [0., 0., 0.]])

One Tensor (2x3):
tensor([[1., 1., 1.],
        [1., 1., 1.]])
```

---

## 3. Tensor Operations: Indexing, Slicing, and Reshaping

Manipulating tensor shapes and sub-regions is an essential skill when building neural network layers (e.g., preparing image batches, flattening feature maps, or reordering dimensions).

### Indexing and Slicing

PyTorch uses standard Python/NumPy indexing syntax:
- `tensor[row_idx]` accesses a specific row.
- `tensor[:, col_idx]` extracts a specific column across all rows.
- `tensor[:n]` extracts the first $n$ rows.

### Reshaping: `.view()` vs. `.reshape()`

When reshaping a tensor (changing its shape without altering its total element count $N = d_1 \times d_2 \times \dots$):

- **`.view(*shape)`**: Returns a new tensor that shares the underlying memory data buffer with the original tensor. It requires the original tensor to be **contiguous** in memory.
- **`.reshape(*shape)`**: Returns a tensor with the requested shape. If the tensor is contiguous, it returns a view; if non-contiguous, it copies the data to a new contiguous memory buffer automatically.

```python
tensor = torch.tensor([[1, 2], [3, 4], [5, 6]])

# Indexing
print("Indexing:")
print("First row:", tensor[0])
print("Second column:", tensor[:, 1])

# Slicing
print("\nSlicing:")
print("First two rows:\n", tensor[:2])

# Reshaping
original_tensor = torch.randint(1, 10, (3, 2))  # Random integers in [1, 9] with shape 3x2
print("\nOriginal Tensor (3x2):")
print(original_tensor)

reshaped_tensor = original_tensor.view(2, 3)  # Reshape to 2x3 using .view()
print("\nReshaped Tensor (2x3 using .view):")
print(reshaped_tensor)

# Reshaping using .reshape()
print("\nReshaping using .reshape():")
print(original_tensor.reshape(2, 3))
```

**Output:**

```text
Indexing:
First row: tensor([1, 2])
Second column: tensor([2, 4, 6])

Slicing:
First two rows:
 tensor([[1, 2],
        [3, 4]])

Original Tensor (3x2):
tensor([[5, 2],
        [9, 8],
        [9, 5]])

Reshaped Tensor (2x3 using .view):
tensor([[5, 2, 9],
        [8, 9, 5]])

Reshaping using .reshape():
tensor([[5, 2, 9],
        [8, 9, 5]])
```

---

## 4. Tensor Functions & Arithmetic

PyTorch provides a rich set of math operations and structural tensor functions.

### Tensor Concatenation (`torch.cat`)

`torch.cat(tensors, dim=0)` joins a sequence of tensors along a specified dimension `dim`:
- **`dim=0`**: Concatenates along rows (stacks vertically, increasing dimension 0).
- **`dim=1`**: Concatenates along columns (stacks horizontally, increasing dimension 1).

All dimensions except `dim` must match between concatenated tensors.

```python
# Integer Tensors
tensor_a = torch.randint(1, 10, (3, 3))
tensor_b = torch.randint(1, 10, (3, 3))

print("Tensor A:")
print(tensor_a)
print("\nTensor B:")
print(tensor_b)

# Concatenating along rows (dim=0) -> output shape (6, 3)
concatenated_rows = torch.cat((tensor_a, tensor_b), dim=0)
print("\nConcatenated along rows (dim=0):")
print(concatenated_rows)

# Concatenating along columns (dim=1) -> output shape (3, 6)
concatenated_cols = torch.cat((tensor_a, tensor_b), dim=1)
print("\nConcatenated along columns (dim=1):")
print(concatenated_cols)
```

**Output:**

```text
Tensor A:
tensor([[2, 9, 9],
        [8, 4, 7],
        [8, 1, 7]])

Tensor B:
tensor([[7, 6, 4],
        [4, 9, 3],
        [9, 6, 5]])

Concatenated along rows (dim=0):
tensor([[2, 9, 9],
        [8, 4, 7],
        [8, 1, 7],
        [7, 6, 4],
        [4, 9, 3],
        [9, 6, 5]])

Concatenated along columns (dim=1):
tensor([[2, 9, 9, 7, 6, 4],
        [8, 4, 7, 4, 9, 3],
        [8, 1, 7, 9, 6, 5]])
```

We can perform the exact same concatenation operation on random floating-point tensors:

```python
tensor_a = torch.rand(3, 3)
tensor_b = torch.rand(3, 3)

print("Concatenated along row (dim=0):\n", torch.cat((tensor_a, tensor_b), dim=0))
print("\nConcatenated along column (dim=1):\n", torch.cat((tensor_a, tensor_b), dim=1))
```

### Element-Wise Arithmetic

Standard arithmetic operators (`+`, `-`, `*`, `/`) perform **element-wise** operations on tensors of matching shapes or broadcastable dimensions.

> **Important Distinction:** `tensor_a * tensor_b` is the **Hadamard (element-wise) product**, whereas `torch.matmul(tensor_a, tensor_b)` or `tensor_a @ tensor_b` is **matrix multiplication**.

```python
tensor_x = torch.tensor([1, 2, 3])
tensor_y = torch.tensor([4, 5, 6])

print("Element-wise Operations:")
print("Addition (+):      ", tensor_x + tensor_y)
print("Subtraction (-):   ", tensor_x - tensor_y)
print("Multiplication (*):", tensor_x * tensor_y)
print("Division (/):      ", tensor_x / tensor_y)
```

**Output:**

```text
Element-wise Operations:
Addition (+):       tensor([5, 7, 9])
Subtraction (-):    tensor([-3, -3, -3])
Multiplication (*): tensor([ 4, 10, 18])
Division (/):       tensor([0.2500, 0.4000, 0.5000])
```

---

## 5. CPU vs. GPU Performance & Memory Profiling

A primary reason PyTorch is widely used for deep learning is its seamless GPU integration via NVIDIA CUDA. Modern GPUs possess thousands of lightweight Arithmetic Logic Units (ALUs) capable of running massive matrix computations in parallel.

### Checking Device Availability

We specify target execution devices using `torch.device`:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### Benchmarking Matrix Addition ($1000 \times 1000$)

Let's compare element-wise matrix addition on a $1000 \times 1000$ floating-point tensor on CUDA vs. CPU.

```python
# GPU execution
torch_size = (1000, 1000)
tensor_a_gpu = torch.rand(torch_size).to(device)
tensor_b_gpu = torch.rand(torch_size).to(device)

print(f"Tensor A Device: {tensor_a_gpu.device}")
print(f"Tensor B Device: {tensor_b_gpu.device}")

# Time GPU matrix addition using Jupyter %timeit
%timeit c_gpu = tensor_a_gpu + tensor_b_gpu
```

**GPU Benchmark Output:**
```text
Using device: cuda
cuda:0
cuda:0
30 μs ± 88.3 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
```

Now let's run the exact same $1000 \times 1000$ matrix addition on CPU:

```python
cpu_tensor_a = torch.rand(torch_size)
cpu_tensor_b = torch.rand(torch_size)

print(f"CPU Tensor A Device: {cpu_tensor_a.device}")
print(f"CPU Tensor B Device: {cpu_tensor_b.device}")

%timeit c_cpu = cpu_tensor_a + cpu_tensor_b
```

**CPU Benchmark Output:**
```text
cpu
cpu
84.5 μs ± 6.42 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
```

For basic element-wise addition, the GPU achieves **$30\ \mu\text{s}$** compared to **$84.5\ \mu\text{s}$** on CPU (~$2.8\times$ speedup).

---

### Benchmarking Matrix Multiplication ($1000 \times 1000$)

Where GPUs truly shine is computationally intensive operations like **matrix multiplication** (`torch.matmul`), which forms the backbone of fully connected layers, convolutions, and transformer attention matrices.

#### GPU Matrix Multiplication (`torch.matmul`)

```python
%timeit c_gpu = torch.matmul(tensor_a_gpu, tensor_b_gpu)
```

**Output:**
```text
328 μs ± 2.04 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
```

#### CPU Matrix Multiplication (`torch.matmul`)

```python
%timeit c_cpu = torch.matmul(cpu_tensor_a, cpu_tensor_b)
```

**Output:**
```text
4.05 ms ± 98 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

### Key Takeaway

| Operation ($1000 \times 1000$) | CPU Execution Time | GPU Execution Time (CUDA) | Speedup Factor |
| :--- | :--- | :--- | :--- |
| **Matrix Addition** | $84.5\ \mu\text{s}$ | $30\ \mu\text{s}$ | **~$2.8\times$** |
| **Matrix Multiplication** | $4.05\ \text{ms}$ ($4050\ \mu\text{s}$) | $328\ \mu\text{s}$ | **~$12.3\times$** |

For matrix multiplication, CUDA GPU execution reduces compute time from **$4.05\ \text{ms}$** down to **$328\ \mu\text{s}$**—an acceleration of over **$12\times$** on a single $1000 \times 1000$ matrix! For larger neural networks with batch sizes in the millions, this hardware acceleration makes deep learning feasible.

---

### Monitoring GPU Memory

PyTorch manages VRAM using an internal **caching memory allocator** to prevent frequent CUDA memory allocation overhead. You can monitor VRAM usage directly with PyTorch CUDA memory utilities:

```python
print("Current GPU Memory Usage:")
print(f"Allocated Memory: {torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB")
print(f"Cached/Reserved Memory: {torch.cuda.memory_reserved(device) / (1024 ** 2):.2f} MB")
```

**Output:**
```text
Current GPU Memory Usage:
Allocated Memory: 20.00 MB
Cached/Reserved Memory: 40.00 MB
```

- **Allocated Memory**: Memory currently occupied by active tensor objects in VRAM.
- **Reserved/Cached Memory**: Memory held by PyTorch's caching allocator from the OS to speed up future tensor allocations.

---

## Summary

In this tutorial, we covered the core building blocks of PyTorch:
1. **Virtual Environments**: Keeping dependencies isolated with `venv` and installing `torch`, `torchvision`, and `torchaudio`.
2. **Tensors**: Creating scalars, vectors, matrices, and multi-dimensional tensors with initializers like `torch.zeros`, `torch.ones`, and `torch.rand`.
3. **Indexing & Reshaping**: Accessing sub-tensors via slices and reshaping dimensions using `.view()` and `.reshape()`.
4. **Operations**: Joining tensors along specified dimensions with `torch.cat()` and applying element-wise arithmetic operators.
5. **GPU Acceleration**: Moving tensors onto CUDA devices with `.to(device)`, measuring significant speedups for matrix operations, and profiling GPU memory usage.

Equipped with these tensor fundamentals, you are ready to explore automatic differentiation (`autograd`), build custom neural network modules with `torch.nn.Module`, and train deep learning models!
