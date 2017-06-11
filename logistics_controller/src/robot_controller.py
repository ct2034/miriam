#!/bin/env python2
import rospy
import numpy as np

print('aaaa')

scale = .5
delta = np.array([6, 4])

def map_to_plan(x_map):
	assert x_map.__class__ == tuple, "Coordinates should be tuples"
	return tuple(np.array(x_map)*scale + delta)

def plan_to_map(x_plan):
	assert x_plan.__class__ == tuple, "Coordinates should be tuples"
	return tuple((np.array(x_plan) - delta) / scale)

rospy.init_node('robot_controller', anonymous=True)
rospy.loginfo("CONTROLER init")

n_robots = rospy.get_param('~n_robots')
rospy.loginfo("n_robots: " + n_robots)
ns_prefix = rospy.get_param('~ns_prefix', 'r')  # This will produce namespaces r1, r2, ...
rospy.loginfo("ns_prefix: " + ns_prefix)

robot_nss = map(lambda i: ns_prefix+str(i+1), range(n_robots))

pose_subscribers = map(lambda ns: rospy.Subscriber(ns+"/logistics_pose", Point, callbackGoal), robot_nss)
