#!/usr/bin/env python2
import imageio
from itertools import product
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import pickle
import sys
import time

from adamsmap.adamsmap import (
    get_random_pos,
    graphs_from_posar,
    init_graph_posar_edgew,
    make_edges,
    eval,
    grad_func,
    fix
    )
# import adamsmap


def optimize(n, ntb, nts, image_fname):
    # Paths
    nn = 2

    # Evaluation
    ne = 50  # evaluation set size

    # The map
    im = imageio.imread(image_fname)

    evalset = np.array([
        [get_random_pos(im),
         get_random_pos(im)]
        for _ in range(ne)])
    evalcosts = []
    evalunsucc = []
    evalbc = []

    alpha = 0.01
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 10E-8

    m_t_p = np.zeros([n, 2])
    v_t_p = np.zeros([n, 2])
    m_t_e = np.zeros([n, n])
    v_t_e = np.zeros([n, n])

    start = time.time()
    for t in range(nts):
        if t == 0:
            posar, edgew = init_graph_posar_edgew(im, n)

        g, ge, pos = graphs_from_posar(n, posar)
        make_edges(n, g, ge, posar, edgew, im)
        e_cost, unsuccesful = eval(t, evalset, nn, g, ge,
                                   pos, posar, edgew, im)
        if t == 0:
            e_cost_initial = e_cost
        print("---")
        ratio = float(t) / nts
        print("%d/%d (%.1f%%)" % (t, nts, 100. * ratio))
        print("Eval cost: %.1f (%-.1f%%)" %
              (e_cost, 100. * (e_cost - e_cost_initial) / e_cost_initial))
        print("N unsuccesful: %d / %d" % (unsuccesful, ne))
        elapsed = time.time() - start
        print("T elapsed: %.1fs / remaining: %.1fs" %
              (elapsed, elapsed/ratio-elapsed if ratio > 0 else np.inf))
        print("edgew min: %.3f / max: %.3f / std: %.3f" %
              (np.min(edgew), np.max(edgew), np.std(edgew)))
        evalcosts.append(e_cost)
        evalunsucc.append(unsuccesful)

        batch = np.array([
            [get_random_pos(im), get_random_pos(im)] for _ in range(ntb)])

        # Adam
        g_t_p, g_t_e, bc_tot = grad_func(
            batch, nn, g, ge, posar, edgew
        )
        bc = float(bc_tot) / batch.shape[0]
        if t == 0:
            b_cost_initial = bc
        print("Batch cost: %.2f (%-.1f%%)" %
              (bc, 100. * (bc - b_cost_initial) / b_cost_initial))
        evalbc.append(bc)

        m_t_p = beta_1*m_t_p + (1-beta_1)*g_t_p
        v_t_p = beta_2*v_t_p + (1-beta_2)*(g_t_p*g_t_p)
        m_cap_p = m_t_p / (1-(beta_1**(t+1)))
        v_cap_p = v_t_p / (1-(beta_2**(t+1)))
        posar_prev = np.copy(posar)
        posar = posar - np.divide(
            (alpha * m_cap_p), (np.sqrt(v_cap_p) + epsilon))
        fix(posar_prev, posar, im)

        m_t_e = beta_1*m_t_e + (1-beta_1)*g_t_e
        v_t_e = beta_2*v_t_e + (1-beta_2)*(g_t_e*g_t_e)
        m_cap_e = m_t_e / (1-(beta_1**(t+1)))
        v_cap_e = v_t_e / (1-(beta_2**(t+1)))
        edgew = edgew - np.divide(
            (alpha * m_cap_e), (np.sqrt(v_cap_e) + epsilon))

    fig = plt.figure()
    plt.plot(evalcosts)
    fig.savefig("res/evalcosts.png")
    fig = plt.figure()
    plt.plot(evalunsucc)
    fig.savefig("res/unsuccesful.png")
    fig = plt.figure()
    plt.plot(evalbc)
    fig.savefig("res/batchcost.png")
    fig = plt.figure(figsize=[8, 8])
    eval(-1, evalset, nn, g, ge, pos, posar, edgew, im)

    store = {
        "evalcosts": evalcosts,
        "batchcost": evalbc,
        "unsuccesful": evalunsucc,
        "posar": posar,
        "edgew": edgew
        }

    with open("res/%s_%d_%d.pkl" % (
            image_fname.split(".")[0].split("/")[-1],
            n,
            nts
    ), "wb") as f:
        pickle.dump(store, f)


if __name__ == "__main__":
    # Training
    ntb = 128  # batch size

    for (image_fname, N, nts) in product(
        [sys.argv[1]],
        [100, 200, 500],
        [4096]
    ):
        optimize(N, ntb, nts, image_fname)
