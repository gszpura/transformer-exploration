from typing import Optional, Iterable, Sized

import numpy as np
from sklearn.datasets import make_blobs, make_moons


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
    def tanh_dv(x: "np.ndarray | Tensor") -> np.ndarray:
        if isinstance(x, Tensor):
            x = x.data
        t = np.tanh(x)
        return Tensor(1 - t**2)

    def squeeze(self):
        return Tensor(self.data.squeeze())


class InputLayer:

    def __init__(self, in_size: int):
        self.a = Tensor(np.random.rand(1, in_size))

    def __call__(self, input_activations: Tensor):
        self.a = input_activations
        return self.a

    def backward(self, **kwargs):
        return self.a.grad

    def parameters(self):
        return []


class FullLayer:

    def __init__(self, in_size: int, out_size: int):
        self.w = Tensor(np.random.rand(in_size, out_size))
        self.b = Tensor(np.random.rand(1, out_size))
        self.z = Tensor(np.random.rand(1, out_size))
        self.a = Tensor(np.random.rand(1, out_size))

    def __call__(self, input_activations: Tensor) -> Tensor:
        self.z = input_activations @ self.w + self.b
        self.a = Tensor.tanh(self.z)
        return self.a

    def backward(self, out_grad: np.ndarray, prev_activations: Tensor):
        """
        out_grad comes from previous layer, for last layer it comes from loss
        for other i-th layer it comes from i+1-th layer

        += is here because of mini-batches of size > 1 where we want to backward multiple times,
        once per example, before we update weights based on the grads

        return of 'self.z.grad @ self.w.data.T' prev_activation.grad which we'll update in another layer
        """
        self.a.grad += out_grad
        self.z.grad += out_grad * Tensor.tanh_dv(self.z.data).data
        self.b.grad += self.z.grad * 1
        self.w.grad += prev_activations.data.T @ self.z.grad
        return self.z.grad @ self.w.data.T

    def parameters(self) -> list[Tensor]:
        return [self.w, self.b]


class NormLayer:

    def __init__(self, in_size: int):
        """
        there are as many a^ as a
        ahat = (a - mi)/sigma

        """
        # self.beta = Tensor(np.random.rand(1, in_size))
        # self.gamma = Tensor(np.random.rand(1, in_size))
        self.ahat = Tensor(np.random.rand(1, in_size))
        self.h = Tensor(np.random.rand(1, in_size))
        self.mi = 0
        self.sigma = 0

    def __call__(self, input_activations: Tensor) -> Tensor:
        a = input_activations
        self.mi = np.mean(a.data)
        self.sigma = np.sqrt(np.mean([(ai - self.mi)**2 for ai in a.data]))
        self.ahat = Tensor(data=(a.data - self.mi)/self.sigma)
        self.h = Tensor.tanh(self.ahat)
        return self.h

    def backward(self, out_grad: np.ndarray, prev_activations: Tensor):
        """
        Heavy derivatives over single pair of input-output nodes:
        symbolic:
        daihat/daj = daihat/dai + daihat/dmi*dmi/daj + daihat/dsigma * dsigma/daj
        values:
        daihat/daj = 1/sigma*delta_ij - 1/(sigma*D) -aihat*ajhat/(sigma*D)
        (fan in from mi and sigma)
        where "i" is index from further layer and "j" is index from input layer

        - there's direct influence from when i=j (delta_ij = 1 when i=j)
        - there indirect influences through mi and sigma when i=j and i!=j

        and for calculating loss correctly we need take into account all "i"s for a given "j"
        so the sums must appear over upper layer neurons/nodes

        so it looks like this:
        dL/daj = 1/sigma * (out_grad*1) - 1/(sigma*D)*sumi(out_grad) - ajhat/(sigma*D) * sumi(out_grad*aihat)
        (fan out)

        derivatives for sigma and mi must be well calculated. Example for mi (second term):

        f' = daihat / dmi * dmi / daj

        daihat / dmi = - 1/sigma
        dmi / daj = 1 / D
        f' = - 1/(sigma*D) - as in the equation above - and later we need sum it over i: out_grad*f'

        similar for sigma derivation but this is more difficult with several stages, results:
        dsigma/daj = 1/2sigma * 2(aj - mi)/D = ajhat/D
        daihat/dsigma = - 1/sigma * aihat
        """
        self.h.grad += out_grad
        hat_grad = out_grad * Tensor.tanh_dv(self.ahat.data).data
        self.ahat.grad += hat_grad
        input_grad = 1/self.sigma * (hat_grad - np.mean(hat_grad) - self.ahat.data*np.mean(hat_grad*self.ahat.data))
        return input_grad

    def parameters(self) -> list[Tensor]:
        return []


class Network:
    """
    Network based on Tensor class

    Tanh and MSE are still hardcoded. TODO: BCE and ReLU implementation
    """

    def __init__(self, layer_sizes: list[int]):
        """
        Layer sizes [2, 4, 4, 1]
        mean there's one output neuron and 2 input neurons and 2 hidden layers with 4 and 4 neurons.
        z and b match the shape of a.

        Best practise is to have all layers 2-dimensional - even if they're just a list of 1dim neurons i.e.
        it is better to have layer shape like: (1, 4) than (4, )
        """
        self.sizes = layer_sizes
        self.layers = [InputLayer(in_size=self.sizes[0])]
        self.input_shape = (1, self.sizes[0])

        for i, shape in enumerate(layer_sizes):
            if i < len(layer_sizes) - 1:
                self.layers.append(FullLayer(layer_sizes[i], layer_sizes[i + 1]))

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
        if a0.shape != self.layers[0].a.data.shape:
            raise Exception("Wrong shapes for input layer")
        input_activation = Tensor(data=a0)
        for layer in self.layers:
            input_activation = layer(input_activation)
        return input_activation

    def get_output(self) -> Tensor:
        return self.layers[-1].a

    def get_result_label(self):
        a = self.layers[-1].a
        return np.argmax(a)

    def mse(self, val: 'np.ndarray | Tensor', true_val: 'np.ndarray | Tensor'):
        """
        MSE for one number:
            (val - true)**2

        MSE for multi-class:
            sum((val - true)**2)/C
        where C is number of classes - this is what np.mean does
        """
        return np.mean((val - true_val) ** 2)

    def mse_dv(self, val: 'np.ndarray | Tensor', true_val: 'np.ndarray | Tensor | float') -> 'np.ndarray | Tensor':
        """
        d(a_final - y)²/da3 = 2*(a_final - y)
        :param val:
        :param true_val:
        :return:
        """
        ln = 1
        if isinstance(true_val, Sized):
            ln = len(true_val)
        return 2 * (val - true_val) / ln

    def backward(self, true_value: np.ndarray):
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
        loss = self.mse(self.get_output().data, true_value)
        out_grad = self.mse_dv(self.get_output().data, true_value) * 1

        # we do not propagate gradients to input layer so we iterate only to 1, not to 0
        for i in range(len(self.sizes))[::-1][:-1]:
            out_grad = self.layers[i].backward(out_grad, self.layers[i - 1].a)
        return loss

    def parameters(self) -> list[Tensor]:
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        params = self.parameters()
        for p in params:
            p.grad = np.zeros_like(p.grad)


def generate_2d_data(n_samples: int = 20, one_hot = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    20 datapoints, 2 features each (matches 2 input size),
    binary classification target 0.0 or 1.0 (matches single output neuron).
    """
    X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    y_onehot = np.eye(2)[y]
    if one_hot:
        y = y_onehot
    return [(row.reshape(1, 2), label) for row, label in zip(X, y)]


def generate_3d_data(
    n_samples: int = 30, n_classes: int = 3, **kwargs
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    30 datapoints, 3 features each (matches 3 input size),
    one-hot target (1, n_classes) (matches n_classes output neurons, for softmax + CE).
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_classes,
        n_features=3,
        cluster_std=1.5,
        random_state=42,
    )
    one_hot = np.eye(n_classes)[y]
    return [
        (row.reshape(1, 3), one_hot[i].reshape(1, n_classes)) for i, row in enumerate(X)
    ]


def sgd(net: Network, data: list[tuple], lr=0.05):
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
            param.data -= lr * param.grad
        total_loss += loss
    return total_loss / len(data)


def gradient_descent(net: Network, data: list[tuple], lr=0.05):
    """
    full grad desc = full batch of all examples
    """
    total_loss = 0
    net.zero_grad()
    for example in data:
        input_values, y = example
        net.forward(input_values)
        loss = net.backward(y)
        total_loss += loss
    # update once
    params = net.parameters()
    for param in params:
        param.data -= lr * param.grad
    return total_loss / len(data)


def eval_net_on_data(net: Network, data: list[tuple[np.ndarray, np.ndarray]]):
    hit = 0.0
    for x, y in data:
        out = net.forward(x)
        # print(out.data, "vs.", y)
        hit += np.argmax(out.data) == np.argmax(y)
    print("accuracy:", hit / len(data))


if __name__ == "__main__":
    """
    Old approach (calculations in the network class):
    Loss at epoch: 0: 1.0656959704649467
    Loss at epoch: 10: 0.0736419422662655
    Loss at epoch: 299: 0.00021114038526854226
    """
    np.random.seed(42)
    net = Network([3, 8, 6, 3])
    data = generate_3d_data(100, one_hot=True)
    train, test = data[:70], data[70:]
    for epoch in range(300):
        loss = sgd(net, train, 0.03)
        # loss = gradient_descent(net, data, 0.03)
        print(f"Loss at epoch: {epoch}: {loss}")
    eval_net_on_data(net, test)
    eval_net_on_data(net, train)
