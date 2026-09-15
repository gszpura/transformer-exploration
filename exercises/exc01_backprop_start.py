"""
dL / dd = f
dd / dc = 1
"""

import math
from typing import Optional

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


class Scalar:
    def __init__(self, v: int | float, name: Optional[str] = None):
        self.data = v
        self.grad = 0
        self._prev = set()
        self._backward = lambda: None
        self.name = name

    def __add__(self, other: "int | float | Scalar"):
        if not isinstance(other, Scalar):
            other = Scalar(other)
        merged_name = f"({self.name or ''}+{other.name or ''})"
        out = Scalar(self.data + other.data, merged_name)
        out._prev = {self, other}

        def _back():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _back

        return out

    def __sub__(self, other: "int | float | Scalar"):
        if not isinstance(other, Scalar):
            other = Scalar(other)
        merged_name = f"({self.name or ''}-{other.name or ''})"
        out = Scalar(self.data - other.data, merged_name)
        out._prev = {self, other}

        def _back():
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _back
        return out

    def __neg__(self):
        return self * (-1)

    def __radd__(self, other: int | float):
        return self + other

    def __rsub__(self, other: int | float):
        return Scalar(other) - self

    def __mul__(self, other: "int | float | Scalar"):
        if not isinstance(other, Scalar):
            other = Scalar(other)
        merged_name = f"({self.name or ''}*{other.name or ''})"
        out = Scalar(self.data * other.data, merged_name)
        out._prev = {self, other}

        def _back():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _back
        return out

    def __truediv__(self, other: "int | float | Scalar"):
        if not isinstance(other, Scalar):
            other = Scalar(other)
        inv = other ** (-1)
        return self.__mul__(inv)

    def __pow__(self, other: "int | float"):
        merged_name = f"({self.name or ''}^{other})"
        out = Scalar(self.data**other, merged_name)
        out._prev = {self}

        def _back():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _back
        return out

    def tanh(self) -> "Scalar":
        h_tan_value = (math.exp(self.data) - math.exp(-self.data)) / (
            math.exp(self.data) + math.exp(-self.data)
        )
        out = Scalar(h_tan_value, f"tanh({self.name or ''})")
        out._prev = {self}

        def _back():
            self.grad += (1 - h_tan_value**2) * out.grad

        out._backward = _back
        return out

    def frepr(self):
        return f"<v:{self.data}/gd:{self.grad}>"

    def freprname(self):
        return f"<v:{self.data}/gd:{self.grad}/name:{self.name or ''}>"

    def __repr__(self):
        return self.frepr()

    def __str__(self):
        return self.__repr__()

    def zero_grad(self):
        self.grad = 0

    def topo(self):
        """
        postorder iterative DFS,
        we visit node first, but add it after children
        nice trick to use the same stack to go back up:
        - queue node again and only later queue children,
          mark True when ready for "up" pass, mark "False" when yet to process on "down" pass
        """
        stack = [(self, False)]
        order = []
        visited = set()
        while stack:
            node, upward_pass = stack.pop()
            if upward_pass:
                order.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                for child in node._prev:
                    stack.append((child, False))
        return order[::-1]

    def backward(self):
        self.grad = 1
        ordered = self.topo()
        for node in ordered:
            node._backward()


class Neuron:
    def __init__(self, w: list[Scalar], b: Scalar):
        self.w = w
        self.b = b
        self.inputs = []
        self.value = Scalar(0)

    def forward(self, inputs: list[Scalar]):
        """
        res = tanh(a*w + b)
        """
        self.inputs = inputs
        self.value = sum(
            [self.inputs[i] * self.w[i] for i in range(len(self.w))], self.b
        ).tanh()
        return self.value

    def backward(self):
        self.value.backward()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, inp_size: int, out_size: int):
        self.neurons: list[Neuron] = [
            Neuron(self._weights(inp_size), Scalar(random.uniform(-1, 1)))
            for i in range(out_size)
        ]

    def _weights(self, inp_size: int):
        return [Scalar(n.item()) for n in np.random.uniform(-1, 1, inp_size)]

    def forward(self, inputs: list[Scalar]):
        res = []
        for neuron in self.neurons:
            res.append(neuron.forward(inputs))
        return res

    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params


class Network:
    def __init__(self, layer_sizes: list[int]):
        self.layers: list[Layer] = []
        for layer_no in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[layer_no], layer_sizes[layer_no + 1]))

    def forward(self, inputs: list[Scalar]) -> list[Scalar]:
        """
        Return raw logits.
        """
        inp = inputs
        for l in self.layers:
            new_inp = l.forward(inp)
            inp = new_inp
        return inp

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        params = self.parameters()
        for p in params:
            p.grad = 0


class MSELoss:
    """
    MSELoss for multiple examples with single neuron output and single y_true value (i.e. 2D setup)
    """

    def __init__(self):
        self.partials: list[Scalar] = []
        self.value = Scalar(0)

    def __call__(self, results: list[Scalar], examples: list[Scalar]):
        self.partials: list[Scalar] = [
            (results[i] - examples[i]) ** 2 for i in range(len(results))
        ]
        self.value = sum(self.partials, Scalar(0)) / Scalar(len(results))
        return self.value

    def backward(self):
        self.value.backward()


def generate_data(n_samples: int = 20) -> list[tuple[list[Scalar], list[Scalar]]]:
    """
    20 datapoints, 2 features each (matches 2 input size),
    binary classification target 0.0 or 1.0 (matches single output neuron).
    """
    X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    return [
        ([Scalar(x) for x in row], [Scalar(float(label))]) for row, label in zip(X, y)
    ]


def visualize_data(data: list[tuple[list[Scalar], list[Scalar]]]):
    X = np.array([[s.data for s in inputs] for inputs, _ in data])
    y = np.array([target[0].data for _, target in data])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.title("make_moons data")
    plt.show()


def neuron_example():
    inputs = [Scalar(3, "a1"), Scalar(2, "a2")]
    weights = [Scalar(0.5, "w1"), Scalar(0.25, "w2")]
    bias = Scalar(10, "b")
    n = Neuron(weights, bias)
    result = n.forward(inputs)
    result.backward()
    print("Topo", result.topo())
    print("=====================================")


def main():
    """
    TODO:
    - implement BCELoss
    - abstract GD and partial GD,
    - can minibatch be implemented with current approach?
    exc02:
    - implement Network with parallel computations and minibatches
    :return:
    """
    net = Network([2, 10, 10, 1])
    lr = 0.02
    epochs = 70
    loss = MSELoss()
    data = generate_data(n_samples=50)
    visualize_data(data)
    for epoch in range(epochs):
        total_loss = 0
        for x, y in data:
            res = net.forward(x)
            loss_value = loss(res, y)
            net.zero_grad()
            loss_value.backward()

            for param in net.parameters():
                param.data -= lr * param.grad
            total_loss += loss_value.data

        print(f"Epoch: {epoch}; loss value:", total_loss / len(data))
    print("PRED:", net.forward([Scalar(0), Scalar(0.25)]))
    print("PRED:", net.forward(data[0][0]), "y:", data[0][1])
    print("PRED:", net.forward(data[10][0]), "y:", data[10][1])


if __name__ == "__main__":
    main()
