"""
dL / dd = f
dd / dc = 1
"""

import math

import numpy as np
import matplotlib.pyplot as plt


class Scalar:

    def __init__(self, v: int | float):
        self.data = v
        self.grad = 0
        self._prev = set()
        self._backward = lambda: None

    def __add__(self, other: 'int | float | Scalar'):
        if not isinstance(other, Scalar):
            other = Scalar(other)
        out = Scalar(self.data + other.data)
        out._prev = {self, other}

        def _back():
            print("_back for:", out.data, "grad:", out.grad)
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _back

        return out

    def __sub__(self, other: 'int | float | Scalar'):
        if not isinstance(other, Scalar):
            other = Scalar(other)
        out = Scalar(self.data - other.data)
        out._prev = {self, other}

        def _back():
            print("_back for:", out.data, "grad:", out.grad)
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _back
        return out

    def __mul__(self, other: 'int | float | Scalar'):
        if not isinstance(other, Scalar):
            other = Scalar(other)
        out = Scalar(self.data * other.data)
        out._prev = {self, other}

        def _back():
            print("_back for:", out.data, "grad:", out.grad)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _back
        return out

    def __repr__(self):
        return f"<v:{self.data}>"

    def __str__(self):
        return self.__repr__()

    def frepr(self):
        return f"<v:{self.data}/gd:{self.grad}>"

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
            node, emitted = stack.pop()
            if emitted:
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
        print(ordered)
        for node in ordered:
            node._backward()


def main():
    a = Scalar(3.0)
    b = Scalar(4.0)
    c = a + b
    d = Scalar(2.0)
    e = c * d
    e.backward()
    t = e.topo()
    print([item.frepr() for item in t])

    print("SECOND")
    x = Scalar(2.0)
    y = x * x
    z = Scalar(3.0)
    L = z + y
    # L.backward()
    print(L.topo())

    print("THIRD")
    x = Scalar(2.0)
    y = Scalar(3.0)
    z = x * y
    L = z + x
    # L.backward()
    print(L.topo())




if __name__ == "__main__":
    main()
