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
        return Tensor(data=self.data @ other.data)

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
        """
        self.w1 = Tensor(np.random.rand(2, 4))
        self.b1 = Tensor(np.random.rand(4))
        self.z1 = Tensor(np.random.rand(4))
        self.w2 = Tensor(np.random.rand(4, 4))
        self.b2 = Tensor(np.random.rand(4))
        self.z2 = Tensor(np.random.rand(4))
        self.w3 = Tensor(np.random.rand(4, 1))
        self.b3 = Tensor(np.random.rand(1))
        self.z3 = Tensor(np.random.rand(1))
        self._acts = [Tensor(np.zeros(4)), Tensor(np.zeros(4)), Tensor(np.zeros(1))]

    def forward(self, a0: np.ndarray):
        self.z1 = self.w1.T @ a0 + self.b1
        a1 = Tensor.tanh(self.z1).squeeze()
        # print(a1.shape, a1, "a1")
        self.z2 = self.w2.T @ a1 + self.b2
        a2 = Tensor.tanh(self.z2).squeeze()
        # print(a2, a2.shape, "a2")
        self.z3 = self.w3.T @ a2 + self.b3
        a3 = Tensor.tanh(self.z3).squeeze()
        self._acts = [Tensor(a0), a1, a2, a3]
        return a3

    def tanh_dv(self, x):
        t = np.tanh(x)
        return 1 - t**2

    def mse(self, val, true_val):
        return (val - true_val)**2

    def mse_dv(self, val, true_val):
        return 2*(val - true_val)

    def backward(self, true_value: float):
        mse = self.mse(self._acts[3].data, true_value)
        # d(a3 - y)²/da3 = 2*(a3 - y)
        self._acts[3].grad = 1*self.mse_dv(self._acts[3].data.item(), true_value)

        # a3, z3, b3, w3
        self.z3.grad = self._acts[3].grad*self.tanh_dv(self.z3.data)
        self.b3.grad = self.z3.grad*1
        self.w3.grad = self.z3.grad*self._acts[2].data.reshape(4, 1)

        # a2, z2, b2
        self._acts[2].grad = self.z3.grad*self.w3.data
        self.z2.grad = self._acts[2].grad*self.tanh_dv(self.z2.data.reshape(4, 1))
        self.b2.grad = self.z2.grad*1

        # NOTE:
        # no need to reshape self._acts and self.z which must match with shapes
        # because we use different way of calc, with for loop
        # TODO: this should be matmul instead of loop, but need to figure it out
        # w2, a1
        w2t = self.w2.T
        for n, _ in enumerate(w2t):
            grad_val = self.z2.grad[n] * self._acts[1].data
            w2t.set_grad_to(n, grad_val)
            self._acts[1].grad += self.z2.grad[n] * w2t[n].data
        # TODO: seems this is hackish, should be revisited
        self.w2 = w2t.T

        # z1, b1
        self.z1.grad = self._acts[1].grad*self.tanh_dv(self.z1.data)
        self.b1.grad = self.z1.grad*1

        # w1, a0
        w1t = self.w1.T
        for n, layer in enumerate(w1t):
            grad_val = self.z1.grad[n] * self._acts[0].data
            w1t.set_grad_to(n, grad_val)
            self._acts[0].grad += self.z1.grad[n] * w1t[n].data
        self.w1 = w1t.T



if __name__ == "__main__":
    np.random.seed(42)
    input_values = np.random.rand(2)
    print("Inputs:", input_values, type(input_values))
    # forward(input_values)
    net = Network()
    res = net.forward(input_values)
    print("RES:", res)
    net.backward(1)
