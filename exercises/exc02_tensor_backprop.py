from typing import Optional

import numpy as np

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

    def set_grad_to(self, key, grad):
        self.grad[key] = grad

    def __iter__(self):
        return self.data.__iter__()

    @staticmethod
    def tanh(item):
        return Tensor(np.tanh(item.data))

    def squeeze(self):
        return Tensor(self.data.squeeze())



class Network:
    """
    TODO:
    - remove for loops from backward
    - make it smooth for any MLP arch
    """

    def __init__(self):
        """
        Predefined size of network layers: [2, 4, 4, 1].
        z and b should match the shape of a
        """
        self.w1 = Tensor(np.random.rand(2, 4))
        self.b1 = Tensor(np.random.rand(1, 4))
        self.z1 = Tensor(np.random.rand(1, 4))
        self.w2 = Tensor(np.random.rand(4, 4))
        self.b2 = Tensor(np.random.rand(1, 4))
        self.z2 = Tensor(np.random.rand(1, 4))
        self.w3 = Tensor(np.random.rand(4, 1))
        self.b3 = Tensor(np.random.rand(1, 1))
        self.z3 = Tensor(np.random.rand(1, 1))
        self._acts = [Tensor(np.zeros(4)), Tensor(np.zeros(4)), Tensor(np.zeros(1))]

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
        a0 = Tensor(a0)
        self.z1 = a0 @ self.w1 + self.b1
        a1 = Tensor.tanh(self.z1)
        self.z2 = a1 @ self.w2  + self.b2
        a2 = Tensor.tanh(self.z2)
        self.z3 = a2 @ self.w3 + self.b3
        a3 = Tensor.tanh(self.z3)
        self._acts = [a0, a1, a2, a3]
        return a3

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

        a2 @ dz3 -> dw3:
        (1, 4)^T @ (1, 1)^T -> (4, 1)

        so standard way: dw = a^T @ dz

        z&b should be the same shape as a!
        """
        loss = self.mse(self._acts[3].data, true_value)
        print("loss:", loss)
        # d(a3 - y)²/da3 = 2*(a3 - y)
        mse_dev = 1 * self.mse_dv(self._acts[3].data.item(), true_value)
        # shape: (1, 1)
        self._acts[3].grad = np.array([mse_dev]).reshape(1, 1)

        # a3, z3, b3, w3
        self.z3.grad = self._acts[3].grad*self.tanh_dv(self.z3.data)
        self.b3.grad = self.z3.grad*1
        # w3 has shape (4, 1): (4, 1) x (1, 1) -> (4, 1)
        self.w3.grad = self._acts[2].data.T @ self.z3.grad

        # a2, z2, b2, w2
        # a2 has shape (1, 4): (1, 1) x (1, 4) -> (1, 4)
        self._acts[2].grad = self.z3.grad @ self.w3.data.T
        # z2 has shape (1, 4): (1, 4) * (1, 4) (note: std mul, not matmul!)
        self.z2.grad = self._acts[2].grad * self.tanh_dv(self.z2.data)
        self.b2.grad = self.z2.grad * 1
        # w2 has shape (4, 4): (4, 1) x (1, 4)
        self.w2.grad = self._acts[1].data.T @ self.z2.grad

        # a1, z1, b1
        self._acts[1].grad = self.z2.grad @ self.w2.data.T
        self.z1.grad = self._acts[1].grad * self.tanh_dv(self.z1.data)
        self.b1.grad = self.z1.grad * 1
        self.w1.grad = self._acts[0].data.T @ self.z1.grad

        # a0
        self._acts[0].grad = self.z1.grad @ self.w1.data.T
        return loss


if __name__ == "__main__":
    np.random.seed(42)
    input_values = np.random.rand(2).reshape(1, 2)
    print("Inputs:", input_values, type(input_values), input_values.shape)
    net = Network()
    res = net.forward(input_values)
    print("RES:", res.data)
    print("backward matrix:")
    net.backward(1)
