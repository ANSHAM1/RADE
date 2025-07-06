# RADE: Reverse AutoDiff Engine

`RADE` is a **from-scratch reverse-mode automatic differentiation engine** written in modern **C++20**, built on top of a custom matrix library. It supports full backpropagation through arbitrary computational graphs using dynamic graph tracing and modular gradient computation.

---

## 🚀 Features

* ✅ Fully custom `Matrix` class for tensor-like operations
* ✅ Reverse-mode AutoDiff engine using dynamically traced computation graphs
* ✅ Seamless matrix math: multiplication, addition, subtraction, and broadcasting
* ✅ Built-in activation functions with automatic differentiation: ReLU, Sigmoid, Tanh
* ✅ Custom loss functions: MSE, BCE, CCE
* ✅ Backward pass via closures for arbitrary graph shapes

---

## 🌱 Motivation

Most autodiff engines are black boxes. RADE aims to be **transparent**, **minimal**, and **educational**, perfect for deep learning students and curious C++ developers who want to understand what happens under the hood of modern ML libraries like PyTorch or TensorFlow.

---

## 🧮 Matrix Module

The `Matrix` class supports:

* Dynamic allocation and dimension tracking
* Initialization via zeros or Xavier-random
* Operator overloading: `+`, `-`, `*`, `/` (matrix-scalar, matrix-matrix)
* Broadcasting-aware activation functions: `relu`, `sigmoid`, `tanh`
* Gradient versions: `dRelu`, `dSigmoid`, `dTanh`
* Loss functions and their derivatives (MSE, BCE, CCE)
* Elementwise and outer products

---

## 🔁 Reverse AutoDiff Engine

The `AutoDiff` module builds a dynamic graph of `Node` objects, where each node:

* Stores its `Matrix` value and gradient
* Tracks its `parents`
* Contains a custom `backward()` closure

During forward pass, operations like `*`, `+`, and `relu()` create nodes with attached backward logic. Calling `Loss.backward()` triggers gradient propagation through the graph.

Supported operations:

* `A * B`, `A + B`, `A - B`
* `elementwiseProduct(A, B)`
* `relu(A)`, `tanh(A)`, `sigmoid(A)`
* `Softmaxed_CCE`, `MSE`, `BCE`, `CCE`

---

## 🧪 Example: Softmax + CCE

```cpp
Matrix Target = Matrix(1, 3, 0);
Target.set(0, 0, 1);

shared_ptr<Node> X1 = make_shared<Node>(Matrix(1, 5, "random"));
shared_ptr<Node> W1 = make_shared<Node>(Matrix(5, 3, "random"));
...
shared_ptr<Node> O1 = relu(X1 * W1 + H1 * U1 + B1);
...
shared_ptr<Node> OUT = O1 + O2;

auto [softmax, loss] = Softmaxed_CCE(OUT, Target);
```

---

## 🧠 Applications

### 📈 Stock Price Regression

RADE was used to build a **custom RNN-based regression model** to predict stock prices from historical market data. It was trained on **5 years of real-world stock time series**.

This demonstrates:

* Real-world usability of RADE's autodiff and matrix engine
* Performance in learning non-linear sequential dependencies
* Minimal external dependencies while maintaining gradient correctness

---

## 🛠 Build Instructions

* Requires **C++20** compiler (MSVC, GCC 12+, Clang 15+)
* Enable modules if needed: `/std:c++20` or `-fmodules-ts`
* Compile all `.ixx` files together

---

## 📌 Highlights

* No external dependencies
* Built from scratch: gradients, losses, activations, forward and backward
* Perfect for neural networks, optimization problems, or as a learning tool

---

## 💬 Author

**Ansham Maurya**

* Email: [anshammaurya2291@gmail.com](mailto:anshammaurya2291@gmail.com)

> "When you know the math and write the graph, you own the intelligence."

---

## 📜 License

This project is licensed under **MIT** — use, modify, learn, or fork freely!
