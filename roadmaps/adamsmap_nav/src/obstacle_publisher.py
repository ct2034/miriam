#!/usr/bin/env python

import rospy, math, tf
from costmap_converter.msg import ObstacleArrayMsg, ObstacleMsg
from geometry_msgs.msg import Polygon, Point32, Quaternion
from nav_msgs.msg import Odometry


class ObstaclePublisher:
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.obstacles = [ObstacleMsg()] * n_agents
        self.received = [False] * n_agents
        self.pubs = []
        for i_a in range(n_agents):
            self.pubs.append(
                rospy.Publisher("/robot_{}/move_base/TebLocalPlannerROS/obstacles".format(i_a), ObstacleArrayMsg, queue_size=1)
            )
            rospy.Subscriber("/robot_{}/odom".format(i_a), Odometry, self.odom_callback, i_a)

    def odom_callback(self, msg, i_a):
        p = Point32()
        p.x = msg.pose.pose.position.x
        p.y = msg.pose.pose.position.y
        om = ObstacleMsg()
        # om.radius = .25
        om.id = i_a

        # orientation between now and last pose
        if len(self.obstacles[i_a].polygon.points):
            yaw = math.atan2(
                p.y - self.obstacles[i_a].polygon.points[0].y,
                p.x - self.obstacles[i_a].polygon.points[0].x
            )
        else:
            rospy.logwarn("no previous message")
            yaw = 0
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)
        om.orientation = Quaternion(*q)
        om.polygon.points = [p]

        om.velocities.twist.linear.x = math.cos(yaw) * msg.twist.twist.linear.x
        om.velocities.twist.linear.y = math.sin(yaw) * msg.twist.twist.linear.x
        om.velocities.twist.angular.z = 0
        self.obstacles[i_a] = om
        self.received[i_a] = True

    def publish_obstacles(self):
        for i_a_pub in range(self.n_agents):
            msg = ObstacleArrayMsg()
            msg.header.frame_id = "map"
            for i_a_odom in range(self.n_agents):
                if i_a_odom != i_a_pub:
                    msg.obstacles.append(self.obstacles[i_a_odom])
            self.pubs[i_a_pub].publish(msg)
        self.received = [False] * n_agents

    def cycle(self):
        while not rospy.is_shutdown():
            while not all(self.received) and not rospy.is_shutdown():
                rospy.sleep(.1)
            self.publish_obstacles()


if __name__ == '__main__':
    rospy.init_node('obstacle_publisher')
    rospy.logdebug("init")

    n_agents = rospy.get_param("~n_agents")
    rospy.loginfo("n_agents: {}".format(n_agents))

    op = ObstaclePublisher(n_agents)
    op.cycle()
