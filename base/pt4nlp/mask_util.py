# -*- coding:utf-8 -*-
# Created by Roger
import torch


def lengths2mask(lengths, max_length, byte=False, negation=False):
    """
    Lengths to Mask, lengths start from 0
    :param lengths:     (batch, )
        tensor([ 1,  2,  5,  3,  4])
    :param max_length:  int, max length
        5
    :param byte:        Return a ByteTensor if True, else a Float Tensor
    :param negation:
        False:
            tensor([[ 1.,  0.,  0.,  0.,  0.],
                    [ 1.,  1.,  0.,  0.,  0.],
                    [ 1.,  1.,  1.,  1.,  1.],
                    [ 1.,  1.,  1.,  0.,  0.],
                    [ 1.,  1.,  1.,  1.,  0.]])
        True:
            tensor([[ 0.,  1.,  1.,  1.,  1.],
                    [ 0.,  0.,  1.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  1.]])
    :return:
        ByteTensor/FloatTensor
    """
    batch_size = lengths.size(0)
    assert max_length >= torch.max(lengths).item()
    assert torch.min(lengths).item() >= 0

    range_i = torch.arange(1, max_length + 1, dtype=torch.long).expand(batch_size, max_length).to(lengths.device)
    batch_lens = lengths.unsqueeze(-1).expand(batch_size, max_length)

    if negation:
        mask = batch_lens < range_i
    else:
        mask = torch.ge(batch_lens, range_i)

    if byte:
        return mask.detach()
    else:
        return mask.float().detach()


def position2mask(position, max_length, byte=False, negation=False):
    """
    Position to Mask, position start from 0
    :param position:     (batch, )
        tensor([ 1,  2,  0,  3,  4])
    :param max_length:  int, max length
        5
    :param byte:        Return a ByteTensor if True, else a Float Tensor
    :param negation:
        False:
            tensor([[ 0.,  1.,  0.,  0.,  0.],
                    [ 0.,  0.,  1.,  0.,  0.],
                    [ 1.,  0.,  0.,  0.,  0.],
                    [ 0.,  0.,  0.,  1.,  0.],
                    [ 0.,  0.,  0.,  0.,  1.]])
        True:
            tensor([[ 1.,  0.,  1.,  1.,  1.],
                    [ 1.,  1.,  0.,  1.,  1.],
                    [ 0.,  1.,  1.,  1.,  1.],
                    [ 1.,  1.,  1.,  0.,  1.],
                    [ 1.,  1.,  1.,  1.,  0.]])
    :return:
        ByteTensor/FloatTensor
    """
    batch_size = position.size(0)

    assert max_length >= torch.max(position).item() + 1
    assert torch.min(position).item() >= 0

    range_i = torch.arange(0, max_length, dtype=torch.long).expand(batch_size, max_length).to(position.device)

    batch_position = position.unsqueeze(-1).expand(batch_size, max_length)

    if negation:
        mask = torch.ne(batch_position, range_i)
    else:
        mask = torch.eq(batch_position, range_i)

    if byte:
        return mask.detach()
    else:
        return mask.float().detach()


def relative_postition2mask(start, end, max_length):
    """
    :param start: Start Position
    :param end:   End Position
    :param max_length: Max Length, so max length must big or equal than max of end
    :return:
    """
    assert torch.max(end).item() <= max_length
    return lengths2mask(end, max_length) - lengths2mask(start, max_length)


def test_split():
    import numpy
    zero = torch.zeros(7).long()
    left_position = torch.from_numpy(numpy.fromiter([0, 3, 4, 5, 4, 7, 5], dtype=numpy.long))
    right_position = torch.from_numpy(numpy.fromiter([3, 4, 6, 7, 8, 9, 8], dtype=numpy.long))
    lens = torch.from_numpy(numpy.fromiter([5, 7, 8, 8, 10, 12, 9], dtype=numpy.long))
    left = relative_postition2mask(zero, left_position, torch.max(lens).item())
    middle = relative_postition2mask(left_position, right_position, torch.max(lens).item())
    right = relative_postition2mask(right_position, lens, torch.max(lens).item())
    print(left)
    print(middle)
    print(right)
    print(left + middle + right)


if __name__ == "__main__":
    test_split()
