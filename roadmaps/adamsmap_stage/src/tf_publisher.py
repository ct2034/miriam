#!/usr/bin/env python2
import rospy
import tf
from nav_msgs.msg import Odometry


def cb(msg, args):
    # msg = Odometry()
    (i_a, tb) = args
    tb.sendTransform(
        translation=[
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ],
        rotation=[
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ],
        time=msg.header.stamp,
        child="/robot_{}/base_link".format(i_a),
        parent="map"
    )
    publish_static_transform(i_a, tb, msg.header.stamp)

def publish_static_transform(i_a, tb, time):
    tb.sendTransform(
        translation=[
            0,
            0,
            0
        ],
        rotation=[
            0,
            0,
            0,
            1
        ],
        time=time,
        child="/robot_{}/odom".format(i_a),
        parent="map"
    )
    tb.sendTransform(
        translation=[
            0,
            0,
            0
        ],
        rotation=[
            0,
            0,
            0,
            1
        ],
        time=time,
        child="robot_{}/base_link_tf2".format(i_a),
        parent="/robot_{}/base_link".format(i_a)
    )


if __name__ == '__main__':
    rospy.init_node('tf_publisher')
    rospy.logdebug("init")

    n_agents = rospy.get_param("~n_agents")
    rospy.loginfo("n_agents: {}".format(n_agents))
    tb = tf.TransformBroadcaster()

    for i_a in range(n_agents):
        rospy.Subscriber("robot_{}/base_pose_ground_truth".format(i_a), Odometry, cb, (i_a, tb))

    rospy.spin()