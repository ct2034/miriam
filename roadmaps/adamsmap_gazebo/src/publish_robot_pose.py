#!/usr/bin/env python2
import numpy as np
import random
import rospy

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest

SET_MODEL_TOPIC_NAME = '/gazebo/set_model_state'
SPAWN_SERVICE_NAME = '/gazebo/spawn_urdf_model'
SIZE = 45
D = 2


def publish_pose(pub, name, pos):
    [x, y] = pos
    ms = ModelState()
    ms.model_name = name
    ms.pose.position.x = x
    ms.pose.position.y = y
    ms.pose.position.z = -1
    pub.publish(ms)


if __name__ == '__main__':
    initialized = False
    n_agents = 10
    model_str = rospy.get_param('/robot_description')
    rospy.init_node('publish_robot_pose')
    pub = rospy.Publisher(SET_MODEL_TOPIC_NAME, ModelState, queue_size=1)
    rospy.wait_for_service(SPAWN_SERVICE_NAME)
    client = rospy.ServiceProxy(SPAWN_SERVICE_NAME, SpawnModel)
    poses = np.zeros([n_agents, 2])
    while(True):
        for i in range(n_agents):
            name = "robot{}".format(i, "%02d")
            if not initialized:
                poses[i, :] = [2 * SIZE * random.random() - SIZE,
                               2 * SIZE * random.random() - SIZE]
                req = SpawnModelRequest()
                req.model_name = name
                req.model_xml = model_str
                res = client(req)
                print(res)
            poses[i, :] = (np.array([D * random.random() - D / 2,
                                     D * random.random() - D / 2])
                           + poses[i, :])
            publish_pose(pub, name, poses[i, :])
        initialized = True
        rospy.sleep(.1)
