#!/usr/bin/env python2
import csv
import numpy as np
import random
import rospy
import sys

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, DeleteModel

SET_MODEL_TOPIC_NAME = '/gazebo/set_model_state'
SPAWN_SERVICE_NAME = '/gazebo/spawn_urdf_model'
DELETE_SERVICE_NAME = '/gazebo/delete_model'
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


def spawn_robot(client, name, model_str):
    req = SpawnModelRequest()
    req.model_name = name
    req.model_xml = model_str
    res = client(req)
    rospy.loginfo(res)


def unspawn_robot(client, name, pub):
    # rospy.loginfo("unspawn " + name)
    # res = client(name)
    # rospy.loginfo(res)
    publish_pose(pub, name, [9999, 9999])



if __name__ == '__main__':
    initialized = False
    model_str = rospy.get_param('/robot_description')
    rospy.init_node('publish_robot_pose')
    pub = rospy.Publisher(SET_MODEL_TOPIC_NAME, ModelState, queue_size=1)
    rospy.wait_for_service(SPAWN_SERVICE_NAME)
    spawn_client = rospy.ServiceProxy(SPAWN_SERVICE_NAME, SpawnModel)
    rospy.wait_for_service(DELETE_SERVICE_NAME)
    delete_client = rospy.ServiceProxy(DELETE_SERVICE_NAME, DeleteModel)
    fname = sys.argv[1]
    paths = []
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            paths.append(list(
                map(float, row)
            ))
    ends = paths[-1]
    T = len(paths) - 1
    n_agents = len(ends) / 2
    backs = []
    for i_a in range(n_agents):
        back = T
        while paths[back][i_a*2:i_a*2+2] == ends[i_a*2:i_a*2+2]:
            back -= 1
        backs.append(back+1)
    t = 0
    for timeslice in paths:
        for i_a in range(n_agents):
            name = "robot{}".format(i_a, "%02d")
            pose = [SCALE * timeslice[i_a*2] + SHIFT,
                    SCALE * timeslice[i_a*2+1] + SHIFT]
            if not initialized:
                spawn_robot(spawn_client, name, model_str)
            if t >= backs[i_a]:
                unspawn_robot(delete_client, name, pub)
            else:
                publish_pose(pub, name, pose)
        initialized = True
        t += 1
        rospy.sleep(.01)
        if(t % 100 == 0):
            rospy.loginfo("{:.1%}".format(float(t) / T))
