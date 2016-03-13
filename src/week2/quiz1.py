from math import exp
from math import fabs


def sigmoid(scalar, w0):
    return 1 / (1 + exp(- scalar * w0))


def likelihood(data):
    likes = [fabs(1 - clazz - sigmoid(value, 1)) for (value, clazz) in data]
    from functools import reduce
    from operator import mul
    print('likelihood:', reduce(mul, likes))


def gradient(data):
    parts = [value * (clazz - sigmoid(value, 1)) for (value, clazz) in data]
    print('gradient:', sum(parts))


def main():
    print('--------quiz--------')
    data = [(2.5, 1), (0.3,0), (2.8, 1), (0.5,1)]
    likelihood(data)
    gradient(data)


if __name__ == '__main__':
    main()
