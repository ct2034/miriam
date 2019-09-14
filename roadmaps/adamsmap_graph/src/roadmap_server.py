#!/usr/bin/env python2
import os

import numpy as np
import matplotlib.pyplot as plt
import rospy
from multiprocessing import Pool
import pickle
import time

from nav_msgs.msg import OccupancyGrid
from graph_msgs.msg import GeometryGraph, Edges
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from adamsmap.adamsmap import (
    get_random_pos,
    graphs_from_posar,
    init_graph_posar_edgew,
    make_edges,
    eval,
    grad_func,
    fix
)


class RoadmapServer:
    info = None

    def __init__(self):
        self.pub_rm = rospy.Publisher("roadmap", GeometryGraph, latch=True, queue_size=1)
        self.pub_rmv = rospy.Publisher("roadmap_viz", MarkerArray, latch=True, queue_size=1)
        self.sub_cm = rospy.Subscriber("/costmap_2d_node/costmap/costmap", OccupancyGrid, self.map_cb)
        self.cache_dir = rospy.get_param("~cache_dir")
        rospy.logdebug("cache_dir: " + self.cache_dir)

        self.ps = None
        self.edges = []
        # self.ps = [
        #     Point(0, 0, 0),
        #     Point(1, 0, 0),
        #     Point(1, 1, 0),
        #     Point(0, 1, 0)
        # ]
        # self.edges = Edges()
        # self.edges.node_ids =
        #     [
        #     [[1],
        #      [3],
        #      [0],
        #      [2]],
        #     [[1., 1.],
        #      [1., 1., 1.],
        #      [1., 1., 1.],
        #      [1., 1.]]
        # ]

    def map_cb(self, map_msg):
        rospy.logdebug("got a costmap")
        self.info = map_msg.info
        rospy.logdebug(self.info)
        assert self.info.width == self.info.height, "currently we need a square map"
        size = self.info.height
        rospy.logdebug(np.histogram(map_msg.data))
        map = np.reshape(map_msg.data, (size, size))

        n = 100
        nts = 1024
        h = hash(map_msg.data)
        fname = self.fname(h, n, nts)
        if os.path.exists(fname):
            rospy.loginfo("found cache file: " + fname)
            with open(fname, "rb") as f:
                store = pickle.load(f)
            posar = store["posar"]
            edgew = store["edgew"]
            im = (map.reshape(map.shape + (1,)) - 100) * (-2.6)
            __, ge, pos = graphs_from_posar(n, posar)
            make_edges(n, __, ge, posar, edgew, im)
            self.store_graph_and_pub(n, ge, posar, edgew)
        else:
            rospy.loginfo("no cache found: " + fname + "\noptimizing....")
            self.optimize(n, 128, nts, map, h)

    def optimize(self, n, ntb, nts, map, hash):
        # Paths
        nn = 1

        # Evaluation
        ne = 128  # evaluation set size

        # The map
        im = (map.reshape(map.shape + (1,)) - 100) * (-2.6)

        # Multiprocessing
        processes = 2  # Number of processes
        # pool = Pool(processes)

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
                                       pos, posar, edgew, im, plot=False)
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
                  (elapsed, elapsed / ratio - elapsed if ratio > 0 else np.inf))
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

            m_t_p = beta_1 * m_t_p + (1 - beta_1) * g_t_p
            v_t_p = beta_2 * v_t_p + (1 - beta_2) * (g_t_p * g_t_p)
            m_cap_p = m_t_p / (1 - (beta_1 ** (t + 1)))
            v_cap_p = v_t_p / (1 - (beta_2 ** (t + 1)))
            posar_prev = np.copy(posar)
            posar = posar - np.divide(
                (alpha * m_cap_p), (np.sqrt(v_cap_p) + epsilon))
            fix(posar_prev, posar, im)

            m_t_e = beta_1 * m_t_e + (1 - beta_1) * g_t_e
            v_t_e = beta_2 * v_t_e + (1 - beta_2) * (g_t_e * g_t_e)
            m_cap_e = m_t_e / (1 - (beta_1 ** (t + 1)))
            v_cap_e = v_t_e / (1 - (beta_2 ** (t + 1)))
            edgew = edgew - np.divide(
                (alpha * m_cap_e), (np.sqrt(v_cap_e) + epsilon))

            self.store_graph_and_pub(n, ge, posar, edgew)

        store = {
            "evalcosts": evalcosts,
            "batchcost": evalbc,
            "unsuccesful": evalunsucc,
            "posar": posar,
            "edgew": edgew
        }

        with open(self.fname(hash, n, nts), "wb") as f:
            pickle.dump(store, f)

    def store_graph_and_pub(self, n, ge, posar, edgew):
        self.ps = []
        for i_p in range(posar.shape[0]):
            self.ps.append(Point(posar[i_p, 0] * .1 - 48,
                                 posar[i_p, 1] * .1 - 48,
                                 0))

        self.edges = []
        for _ in range(n):
            self.edges.append(Edges() )
        for e in ge.edges:
            self.edges[e[0]].node_ids.append(e[1])
            self.edges[e[0]].weights.append(edgew[e[0], e[1]])
        self.publish_viz()
        self.publish_graph()

    def fname(self, hash, n, nts):
        return (self.cache_dir +
        "/%s_%d_%d.pkl" % (
            hash,
            n,
            nts
        ))

    def publish_viz(self):
        ma = MarkerArray()
        id = 0
        for v, edges in enumerate(self.edges):
            for to in edges.node_ids:
                arrow = Marker()
                arrow.id = id
                id += 1
                arrow.header.frame_id = "map"
                arrow.pose.orientation.w = 1
                arrow.scale.x = .05
                arrow.scale.y = .15
                arrow.color.a = .7
                arrow.color.b = 1
                arrow.type = Marker.ARROW
                arrow.points = [
                    self.ps[v],
                    self.ps[to]
                ]
                ma.markers.append(arrow)
        for p in self.ps:
            point = Marker()
            point.id = id
            id += 1
            point.header.frame_id = "map"
            point.pose.orientation.w = 1
            point.scale.x = .3
            point.scale.y = .3
            point.scale.z = .3
            point.color.a = .7
            point.color.r = 1
            point.type = Marker.SPHERE
            point.pose.position.x = p.x
            point.pose.position.y = p.y
            ma.markers.append(point)

        self.pub_rmv.publish(ma)

    def publish_graph(self):
        gg = GeometryGraph()
        gg.nodes = self.ps
        gg.edges = self.edges
        self.pub_rm.publish(gg)


if __name__ == '__main__':
    rospy.init_node('roadmap_server')
    rospy.logdebug("init")

    rs = RoadmapServer()

    while rs.info is None and not rospy.is_shutdown():  # waiting for first map msg
        rospy.sleep(.1)
    rospy.logdebug("got the first map")

    while not rospy.is_shutdown():
        if rs.ps is not None:
            rs.publish_viz()
        rospy.sleep(1)

    rospy.spin()
