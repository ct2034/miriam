#!/usr/bin/env python2
import csv
import numpy as np
import random
import rospy
import sys

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest

SET_MODEL_TOPIC_NAME = '/gazebo/set_model_state'
SPAWN_SERVICE_NAME = '/gazebo/spawn_urdf_model'
SCALE = .1
SHIFT = -48


def publish_pose(pub, name, pos):
    [x, y] = pos
    ms = ModelState()
    ms.model_name = name
    ms.pose.position.x = x
    ms.pose.position.y = -y
    ms.pose.position.z = -.5
    pub.publish(ms)


if __name__ == '__main__':
    initialized = False
    model_str = rospy.get_param('/robot_description')
    rospy.init_node('publish_robot_pose')
    pub = rospy.Publisher(SET_MODEL_TOPIC_NAME, ModelState, queue_size=1)
    rospy.wait_for_service(SPAWN_SERVICE_NAME)
    client = rospy.ServiceProxy(SPAWN_SERVICE_NAME, SpawnModel)
    fname = sys.argv[1]
    with open(fname, 'r') as f :
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            if not initialized:
                n_agents = len(row) / 2
            for i_a in range(n_agents):
                name = "robot{}".format(i_a, "%02d")
                pose = [SCALE * float(row[i_a*2]) + SHIFT,
                        SCALE * float(row[i_a*2+1]) + SHIFT]
                if not initialized:
                    req = SpawnModelRequest()
                    req.model_name = name
                    req.model_xml = model_str
                    res = client(req)
                    print(res)
                publish_pose(pub, name, pose)
            initialized = True
            rospy.sleep(.01)
