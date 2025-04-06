# 1: Linear Regression with Multiple Variables

## 🎯 Learning Objectives

- Understand multivariable linear regression
- Use vectorized operations with numpy
- Compute the cost function \( J(\mathbf{w}, b) \)
- Implement gradient descent for multiple variables

---

## 🧠 1. Problem Setup

You have a dataset with **n features**:

\[
\left\{ (\mathbf{x}^{(1)}, y^{(1)}), (\mathbf{x}^{(2)}, y^{(2)}), \dots, (\mathbf{x}^{(m)}, y^{(m)}) \right\}
\]

Where:
- \( \mathbf{x}^{(i)} \in \mathbb{R}^n \): feature vector for example \( i \)
- \( y^{(i)} \in \mathbb{R} \): output value for example \( i \)
- \( m \): number of training examples
- \( n \): number of features

---

## 📐 2. Model (Hypothesis Function)

The model predicts output as:

\[
f_{\mathbf{w}}(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b = \sum_{j=0}^{n-1} w_j x_j + b
\]

Where:
- \( \mathbf{w} = [w_0, w_1, \dots, w_{n-1}] \in \mathbb{R}^n \)
- \( \mathbf{x} = [x_0, x_1, \dots, x_{n-1}] \in \mathbb{R}^n \)

---

## 💰 3. Cost Function

\[
J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left(f_{\mathbf{w}}(\mathbf{x}^{(i)}) - y^{(i)}\right)^2
\]

---

## ⚙️ 4. Gradient Descent

We compute partial derivatives:

\[
\frac{\partial J(\mathbf{w}, b)}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left(f_{\mathbf{w}}(\mathbf{x}^{(i)}) - y^{(i)}\right) \cdot x_j^{(i)}
\]

\[
\frac{\partial J(\mathbf{w}, b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left(f_{\mathbf{w}}(\mathbf{x}^{(i)}) - y^{(i)}\right)
\]

Update rules:
\[
w_j := w_j - \alpha \cdot \frac{\partial J}{\partial w_j}
\quad \text{for each } j = 0 \dots n-1
\]

\[
b := b - \alpha \cdot \frac{\partial J}{\partial b}
\]

---
#  2: Feature Scaling, Learning Rate, and Feature Engineering

---

## 📏 1. Feature Scaling (Normalization)

### ❓ Why Feature Scaling?

- Features with very different scales (e.g., sqft vs bedrooms) can slow down gradient descent or cause convergence issues.
- Helps gradient descent move efficiently across all dimensions.

### 🔢 Method: Mean Normalization (Standard Score)

For each feature \( x_j \):

\[
x_j^{(i)} := \frac{x_j^{(i)} - \mu_j}{\sigma_j}
\]

Where:
- \( \mu_j \): mean of feature \( j \)
- \( \sigma_j \): standard deviation of feature \( j \)

### ✅ Python Example

```python
# Normalize training data
x_mean = np.mean(x_train, axis=0)
x_std = np.std(x_train, axis=0)
x_norm = (x_train - x_mean) / x_std

