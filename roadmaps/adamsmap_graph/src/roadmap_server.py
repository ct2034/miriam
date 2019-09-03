#!/usr/bin/env python2
import numpy as np
import matplotlib.pyplot as plt
import rospy
from nav_msgs.msg import OccupancyGrid
from graph_msgs.msg import GeometryGraph
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
        self.pub_rmv = None
        self.pub_rm = rospy.Publisher("roadmap", GeometryGraph)
        self.pub_rmv = rospy.Publisher("roadmap_viz", MarkerArray)
        self.sub_cm = rospy.Subscriber("/costmap_2d_node/costmap/costmap", OccupancyGrid, self.map_cb)

        self.ps = [
            Point(0, 0, 0),
            Point(1, 0, 0),
            Point(1, 1, 0),
            Point(0, 1, 0)
        ]
        self.edges = [
            [[1],
             [3],
             [0],
             [2]],
            [[1., 1.],
             [1., 1., 1.],
             [1., 1., 1.],
             [1., 1.]]
        ]

    def map_cb(self, map_msg):
        rospy.logdebug("got a costmap")
        self.info = map_msg.info
        rospy.logdebug(self.info)
        assert self.info.width == self.info.height
        size = self.info.height
        rospy.loginfo(np.histogram(map_msg.data))
        map = np.reshape(map_msg.data, (size, size))

        rospy.logdebug("occupied: " + str(map[0, 0]))
        rospy.logdebug("free: " + str(map[int(size / 2), int(size / 2)]))

        gg = GeometryGraph()
        gg.nodes = self.ps
        gg.edges = self.edges
        self.pub_rm.publish(gg)

    def publish_viz(self):
        while self.pub_rmv is None:
            rospy.sleep(1)

        ma = MarkerArray()
        id = 0
        for v, edges in enumerate(self.edges[0]):
            for to in edges:
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


if __name__ == '__main__':
    rospy.init_node('roadmap_server', log_level=rospy.DEBUG)
    rospy.logdebug("init")

    rs = RoadmapServer()

    while rs.info is None and not rospy.is_shutdown():  # waiting for first map msg
        rospy.sleep(.1)
    rospy.logdebug("got the first map")

    while not rospy.is_shutdown():
        rs.publish_viz()
        rospy.sleep(1)

    rospy.spin()