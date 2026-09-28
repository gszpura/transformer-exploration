from typing import Optional

import numpy as np
from sklearn.datasets import make_moons

class Tensor:

    def __init__(self, data: np.ndarray, grad: Optional[np.ndarray] = None):
        self.data = data
        self.grad = grad if grad is not None else np.zeros_like(self.data)

    @property
    def T(self):
        return Tensor(data=self.data.T, grad=self.grad.T)

    def __matmul__(self, other):
        if isinstance(other, np.ndarray):
            other = Tensor(other)
        if isinstance(other, float):
            other = Tensor(np.array(other))
        return Tensor(data=self.data @ other.data)

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            other = Tensor(other)
        if isinstance(other, (float, int)):
            other = Tensor(np.array(other))
        return Tensor(data=self.data * other.data)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        return Tensor(data=self.data + other.data)

    def __getitem__(self, key):
        return Tensor(data=self.data[key], grad=self.grad[key])

    def __iter__(self):
        return self.data.__iter__()

    @staticmethod
    def tanh(item):
        return Tensor(np.tanh(item.data))

    @staticmethod
    def tanh_dv(x: 'np.ndarray | Tensor') -> np.ndarray:
        if isinstance(x, Tensor):
            x = x.data
        t = np.tanh(x)
        return Tensor(1 - t**2)

    def squeeze(self):
        return Tensor(self.data.squeeze())



class Network:
    """
    Network based on Tensor class
    """

    def __init__(self, layer_sizes: list[int]):
        """
        Layer sizes [2, 4, 4, 1]
        mean there's one output neuron and 2 input neurons and 2 hidden layers with 4 and 4 neurons.
        z and b match the shape of a.

        Best practise is to have all layers 2-dimensional - even if they're just a list of 1dim neurons i.e.
        it is better to have layer shape like: (1, 4) than (4, )
        """
        self.w = []
        self.b = []
        self.z = []
        self.a = []
        self.sizes = layer_sizes

        for i, shape in enumerate(layer_sizes):
            if i < len(layer_sizes) - 1:
                self.w.append(Tensor(np.random.rand(layer_sizes[i], layer_sizes[i + 1])))
            self.b.append(Tensor(np.random.rand(1, layer_sizes[i])))
            self.z.append(Tensor(np.random.rand(1, layer_sizes[i])))
            self.a.append(Tensor(np.random.rand(1, layer_sizes[i])))

    def forward(self, a0: np.ndarray):
        """
        a:
        (1, 2), (1, 4), (1, 4), (1, 1)
        w:
        (2, 4), (4, 4), (4, 1)

        forward (a x w + b)
        (1, 2) x (2, 4) -> (1, 4)
        (1, 4) x (4, 4) -> (1, 4)
        (1, 4) x (4, 1) -> (1, 1)
        """
        if a0.shape != self.a[0].data.shape:
            raise Exception("Wrong shapes for input layer")

        self.a[0].data = a0

        for i in range(len(self.sizes))[1:]:
            # z1 = a0 @ w0 + b1
            # a1 = tanh(z1)
            self.z[i] = self.a[i - 1] @ self.w[i - 1] + self.b[i]
            self.a[i] = Tensor.tanh(self.z[i])
        return self.a[-1]

    def tanh_dv(self, x: np.ndarray) -> np.ndarray:
        t = np.tanh(x)
        return 1 - t**2

    def mse(self, val, true_val):
        return (val - true_val)**2

    def mse_dv(self, val, true_val):
        return 2*(val - true_val)

    def backward(self, true_value: float):
        """
        a, b, z:
        (1, 2), (1, 4), (1, 4), (1, 1)
        w:
        (2, 4), (4, 4), (4, 1)

        forward
        (1, 2) x (2, 4) -> (1, 4)
        (1, 4) x (4, 4) -> (1, 4)
        (1, 4) x (4, 1) -> (1, 1)

        backward
        dz = da * tanh_dv(z)
        dw = a^T @ dz
        """
        loss = self.mse(self.a[-1].data, true_value)
        # print("loss:", loss)
        # d(a3 - y)²/da3 = 2*(a3 - y)
        mse_dev = 1 * self.mse_dv(self.a[-1].data.item(), true_value)
        # shape: (1, 1)
        self.a[-1].grad = np.array([mse_dev]).reshape(1, 1)

        # we do not propagate gradients to input layer so we iterate only to 1 not to 0
        for i in range(len(self.sizes))[::-1][:-1]:
            # each component has prev layer grad and derivative w.r.t. its node
            self.z[i].grad = self.a[i].grad * Tensor.tanh_dv(self.z[i].data).data
            self.b[i].grad = self.z[i].grad * 1
            self.w[i-1].grad = self.a[i-1].data.T @ self.z[i].grad
            self.a[i-1].grad = self.z[i].grad @ self.w[i-1].data.T
        return loss

    def parameters(self) -> list[Tensor]:
        params = []
        for w in self.w:
            params.append(w)
        for b in self.b:
            params.append(b)
        return params

    def zero_grad(self):
        params = self.parameters()
        for p in params:
            p.grad = np.zeros_like(p.grad)



def generate_2d_data(n_samples: int = 20) -> list[tuple[np.ndarray, float]]:
    """
    20 datapoints, 2 features each (matches 2 input size),
    binary classification target 0.0 or 1.0 (matches single output neuron).
    """
    X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    return [(row.reshape(1, 2), label) for row, label in zip(X, y)]


def sgd(net: Network, data: list[tuple], lr = 0.05):
    """
    mini-batch of size 1
    """
    total_loss = 0
    for example in data:
        input_values, y = example
        net.forward(input_values)
        net.zero_grad()
        loss = net.backward(y)
        # update
        params = net.parameters()
        for param in params:
            param.data -= lr*param.grad
        total_loss += loss
    return total_loss / len(data)


def gradient_descent(net: Network, data: list[tuple], lr = 0.05):
    """
    full grad desc = full batch of all examples
    """
    total_loss = 0
    for example in data:
        input_values, y = example
        net.forward(input_values)
        net.zero_grad()
        loss = net.backward(y)
        total_loss += loss
    # update once
    params = net.parameters()
    for param in params:
        param.data -= lr * param.grad
    return total_loss / len(data)


if __name__ == "__main__":
    np.random.seed(42)
    net = Network([2, 8, 6, 1])
    data = generate_2d_data(20)
    for epoch in range(300):
        loss = sgd(net, data, 0.03)
        # loss = gradient_descent(net, data, 0.05)
        print(f"Loss at epoch: {epoch}: {loss}")
