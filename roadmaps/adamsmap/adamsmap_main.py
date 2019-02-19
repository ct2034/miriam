
import imageio
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

from adamsmap import (
    get_random_pos,
    init_graph_posar,
    graph_from_posar,
    make_edges,
    eval,
    grad_func,
    fix
    )


if __name__ == "__main__":
    # Graph
    N = 800

    # Paths
    nn = 3
    MAX_COST = 100000

    # Training
    ntb = 100  # batch size
    nts = 500  # number of batches

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

    alpha = 0.1
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 10E-8

    m_t = np.zeros([N, 2])
    v_t = np.zeros([N, 2])

    start = time.time()
    for t in range(nts):
        if t == 0:
            posar = init_graph_posar(im, N)
        g, pos = graph_from_posar(N, posar)
        make_edges(N, g, posar, im)
        e_cost, unsuccesful = eval(t, evalset, nn, g, pos, posar, im)
        if t == 0:
            e_cost_initial = e_cost
        print("---")
        ratio = float(t / nts)
        print("%d/%d (%.1f%%)" % (t, nts, 100. * ratio))
        print("Eval cost: %.1f (%-.1f%%)" %
              (e_cost, 100. * (e_cost - e_cost_initial) / e_cost_initial))
        print("N unsuccesful: " + str(unsuccesful))
        elapsed = time.time() - start
        print("T elapsed: %.1fs / remaining: %.1fs" %
              (elapsed, elapsed/ratio-elapsed if ratio > 0 else np.inf))
        evalcosts.append(e_cost)
        evalunsucc.append(unsuccesful)

        batch = np.array([
            [get_random_pos(im), get_random_pos(im)] for _ in range(ntb)])
        # Adam
        # ~~~~
        g_t = grad_func(posar, batch, nn, g, posar)
        m_t = beta_1*m_t + (1-beta_1)*g_t
        v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t)
        m_cap = m_t / (1-(beta_1**(t+1)))
        v_cap = v_t / (1-(beta_2**(t+1)))
        posar_prev = np.copy(posar)
        posar = posar - np.divide((alpha * m_cap), (np.sqrt(v_cap) + epsilon))
        fix(posar_prev, posar, im)

    fig = plt.figure()
    plt.plot(evalcosts)
    fig.savefig("costs.png")
    fig = plt.figure()
    plt.plot(evalunsucc)
    fig.savefig("unsuccesful.png")
    fig = plt.figure(figsize=[8, 8])
    eval(-1, evalset, nn, g, pos, posar, im)
