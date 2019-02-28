#!python3
import imageio
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

from adamsmap import (
    get_random_pos,
    init_graph_posar_edgew,
    graphs_from_posar,
    make_edges,
    eval,
    grad_func,
    fix
    )


if __name__ == "__main__":
    # Graph
    N = 200

    # Paths
    nn = 3
    MAX_COST = 100000

    # Training
    ntb = 256  # batch size
    nts = 1024  # number of batches

    # Evaluation
    ne = 50  # evaluation set size

    im = imageio.imread(sys.argv[1])
    im_shape = im.shape

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

    m_t_p = np.zeros([N, 2])
    v_t_p = np.zeros([N, 2])
    m_t_e = np.zeros([N, N])
    v_t_e = np.zeros([N, N])

    start = time.time()
    for t in range(nts):
        if t == 0:
            posar, edgew = init_graph_posar_edgew(im, N)
        g, ge, pos = graphs_from_posar(N, posar)
        make_edges(N, g, ge, posar, edgew, im)
        e_cost, unsuccesful = eval(t, evalset, nn, g, ge, pos, posar, edgew, im)
        if t == 0:
            e_cost_initial = e_cost
        print("---")
        ratio = float(t / nts)
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
        g_t_p, g_t_e, bc_tot = grad_func(batch, nn, g, ge, posar, edgew)
        bc = bc_tot / batch.shape[0]
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
    fig.savefig("evalcosts.png")
    fig = plt.figure()
    plt.plot(evalunsucc)
    fig.savefig("unsuccesful.png")
    fig = plt.figure()
    plt.plot(evalbc)
    fig.savefig("batchcost.png")
    fig = plt.figure(figsize=[8, 8])
    eval(-1, evalset, nn, g, ge, pos, posar, edgew, im)
