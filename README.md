# Introduction to PyTorch workshop

Welcome to the PyTorch workshop repository. In this workshop, we will build and train a neural network for binary classification using the Make Moons dataset.

The goal of this workshop is to provide hands-on experience with:

* Neural network implementation in PyTorch
* Dataset and DataLoader abstractions
* Training loops and optimization
* Binary classification workflows and evaluation

---

# Workshop Structure

The repository is organized into the following tasks:

| File | Task | Task Description |
| --- | --- | --- |
| `model.py`   | Task 1     | Define the neural network architecture     |
| `dataset.py` | Task 2     | Create the dataset and DataLoader pipeline |
| `train.py`   | Task 3 & 4 | Implement training and evaluation loops    |

Template code has already been provided in each file. During the workshop, you will be completing the implementation step-by-step.

---

# Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/venkatKrishnan86/pytorch-workshop.git
cd pytorch-workshop/code
```

---

## 2. Install Dependencies

This project uses `uv` for dependency management.

Install all the required dependencies using:

```bash
uv sync
```

This will automatically create the environment and install the required packages.
To use these packages, you will have to activate the created environment:

```bash
source .venv/bin/activate
```

---

# Running the Code

After syncing dependencies and activating the environment, you can run the training script as usual:

```bash
python train.py
```

---

# Workshop Objectives

By the end of this workshop, participants should be able to:

* Understand the workflow of training neural networks in PyTorch
* Work with tensors, datasets, and dataloaders
* Implement forward passes and loss computation
* Train a binary classifier using gradient descent and backpropagation
* Interpret decision boundaries on nonlinear datasets

---

# Recommended Background

Basic familiarity with:

* Python programming
* Linear algebra fundamentals

No prior PyTorch experience is required.

---

# Dataset

We use the classic **Make Moons** dataset, a synthetic nonlinear binary classification dataset commonly used to demonstrate:

* Decision boundaries
* Nonlinear separability
* Neural network learning behavior
