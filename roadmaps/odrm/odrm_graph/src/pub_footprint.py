#!/usr/bin/env python2
from math import sqrt

import rospy
from geometry_msgs.msg import Point32, Polygon

if __name__ == "__main__":
    rospy.init_node("pub_footprint")
    rospy.logdebug("init")
    pub = rospy.Publisher("/costmap_2d_node/costmap/footprint", Polygon, queue_size=10)
    radius = rospy.get_param("~radius")
    rospy.logdebug("radius: {}".format(radius))
    corner = radius / sqrt(2)

    while not rospy.is_shutdown():  # waiting for first map msg
        rospy.sleep(0.1)
        p = Polygon(
            [
                Point32(-corner, -corner, 0),
                Point32(-corner, corner, 0),
                Point32(corner, corner, 0),
                Point32(corner, -corner, 0),
            ]
        )
        pub.publish(p)
