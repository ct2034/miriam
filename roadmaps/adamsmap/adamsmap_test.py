#!python3
from itertools import combinations
from math import sqrt
import networkx as nx
import numpy as np
from random import random

from adamsmap import (
    grad_func,
    graphs_from_posar,
    make_edges
    )


def test_grad_func():
    # finite-difference (FD) approximation
    # https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/
    N = 25
    nn = 3
    g = nx.DiGraph()
    g.add_nodes_from(range(N))
    posar = np.zeros([N, 2])
    for i in range(N):
        d = float(i%sqrt(N))
        posar[i, :] = [d + .1 * random(),
                       (i-d)/sqrt(N) + .1 * random()]
    edgew = np.triu(np.random.normal(loc=0, scale=0.5, size=(N, N)), 1)
    g, ge, pos = graphs_from_posar(N, posar)
    im = np.ones([800,800,4]) * 255
    make_edges(N, g, ge, posar, edgew, im)
    batch = np.array(
        [[posar[i]+.5,
          posar[j]+.5]
          for (i, j) in [(0, 18)]]#combinations(range(1), 2)]
        )
    dp, de, c = grad_func(batch, nn, g, ge, posar, edgew)
    assert c > 0
    assert(np.all(de == np.triu(de, 1)))
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    epsilon = 1E-6
    # EDGE WEIGTHS
    comp_e = np.zeros(shape=edgew.shape)
    for i in range(N):
        for j in range(N):
            d = np.zeros(shape=edgew.shape)
            d[i, j] = 1
            dpp, dep, cp = grad_func(batch, nn, g, ge, posar, edgew + d * epsilon)
            dpm, dem, cm = grad_func(batch, nn, g, ge, posar, edgew - d * epsilon)
            comp_e[i, j] = (cp - cm) / 2 / epsilon
    assert(np.max(np.abs(de - comp_e)) < 1E-4)
    # POSITIONS
    comp_p = np.zeros(shape=posar.shape)
    for i in range(N):
        for j in range(2):
            d = np.zeros(shape=posar.shape)
            d[i, j] = 1
            dpp, dep, cp = grad_func(batch, nn, g, ge, posar + d * epsilon, edgew)
            dpm, dem, cm = grad_func(batch, nn, g, ge, posar - d * epsilon, edgew)
            comp_p[i, j] = (cp - cm) / 2 / epsilon
    print(dp)
    print(comp_p)
    assert(np.max(np.abs(dp - comp_p)) < 1E-4)

if __name__ == '__main__':
    test_grad_func()
