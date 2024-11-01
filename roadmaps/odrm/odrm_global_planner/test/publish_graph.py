#!/usr/bin/env python2
import rospy
from graph_msgs.msg import Edges, GeometryGraph
from geometry_msgs.msg import Point

if __name__ == "__main__":
    rospy.init_node("pubish_graph", log_level=rospy.INFO)

    rospy.loginfo("Publishing the example graph.")
    pub = rospy.Publisher("roadmap", GeometryGraph, queue_size=1, latch=True)

    exmaple_gg = GeometryGraph()
    exmaple_gg.header.frame_id = "map"
    exmaple_gg.header.stamp = rospy.Time.now()

    exmaple_gg.nodes = [
        Point(1, 3, 0),  # 0
        Point(3, 3, 0),
        Point(2, 2, 0),  # 2
        Point(1, 1, 0),
        Point(3, 1, 0),  # 4
    ]
    node_ids = [[1], [4], [0, 1], [0, 2], [3, 2]]
    weights = [[2], [2], [1.414, 1.414], [2, 1.414], [2, 1.414]]
    for i in range(len(exmaple_gg.nodes)):
        e = Edges()
        e.node_ids = node_ids[i]
        e.weights = weights[i]
        exmaple_gg.edges.append(e)

    r = rospy.Rate(0.1)
    while not rospy.is_shutdown():
        pub.publish(exmaple_gg)
        r.sleep()
        rospy.logdebug("The example graph was published.")
    rospy.spin()
