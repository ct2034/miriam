#!/usr/bin/env python2
import rospy
from geometry_msgs.msg import Polygon, Point32

info = None

if __name__ == '__main__':
    rospy.init_node('pub_footprint')
    rospy.logdebug("init")
    pub = rospy.Publisher("/costmap_2d_node/costmap/footprint", Polygon, queue_size=10)

    while not rospy.is_shutdown():  # waiting for first map msg
        rospy.sleep(.1)
        p = Polygon([
            Point32(-1, -1, 0),
            Point32(-1, 1, 0),
            Point32(1, 1, 0),
            Point32(1, -1, 0)
        ])
        pub.publish(p)
