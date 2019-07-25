#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Code mostly from https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/10
# Created by Roger
import torch


class GradReverse(torch.autograd.Function):

    def __init__(self, lamb):
        self.lamb = lamb

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return grad_output * -self.lamb


class GradReverseLayer(torch.nn.Module):

    def __init__(self, lamb):
        super(GradReverseLayer, self).__init__()
        self.lamb = lamb

    def extra_repr(self):
        return 'lamb={}'.format(self.lamb)

    def forward(self, x):
        return GradReverse(lamb=self.lamb)(x)


def grad_reverse(x, lamb):
    return GradReverse(lamb)(x)


def test():
    x = torch.FloatTensor(5, 5).uniform_()

    f = torch.nn.Linear(5, 1)

    print('Standard')
    y = torch.sum(f(x))
    y.backward()
    y_grad = f.weight.grad
    print(y_grad)
    print(y)
    f.weight.grad = None

    print('Grad Reverse with 1')
    grad_reverse_layer = GradReverseLayer(1)
    y_reverse = torch.sum(grad_reverse_layer(f(x)))
    y_reverse.backward()
    y_reverse_grad = f.weight.grad
    print(y_reverse_grad)
    print(y_reverse)

    f.weight.grad = None

    print('Grad Scale with 0.5')
    grad_scale_layer = GradReverseLayer(-0.5)
    y_scale = torch.sum(grad_scale_layer(f(x)))
    y_scale.backward()
    y_scale_grad = f.weight.grad
    print(y_scale_grad)
    print(y_scale)

    assert torch.equal(y_reverse, y_scale)
    assert torch.equal(y_reverse_grad, -2 * y_scale_grad)


if __name__ == "__main__":
    test()
