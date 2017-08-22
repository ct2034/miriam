#!/bin/env python2
import sys
import numpy as np
import cherrypy

PY2 = sys.version_info[0] == 2
if PY2:
    import rospy
    from geometry_msgs.msg import Pose2D

scale = .5
delta = np.array([6, 4])


def callback_pose(self, data):
    rospy.loginfo("received pose")
    pose = data


if PY2:
    class Interface:
        rospy.init_node('robot_controller', anonymous=True)
        rospy.loginfo("CONTROLLER init")
        initialized = False

        n_robots = rospy.get_param('~n_robots')
        rospy.loginfo("n_robots: " + str(n_robots))
        # This will produce namespaces r1, r2, ...
        ns_prefix = rospy.get_param('~ns_prefix', 'r')
        rospy.loginfo("ns_prefix: " + ns_prefix)

        robot_nss = [ns_prefix + str(i) for i in range(n_robots)]

        pose = None

        pose_subscribers = map(lambda ns: rospy.Subscriber(
            ns + "/logistics_pose", Pose2D, callback_pose), robot_nss)

        @cherrypy.expose
        @cherrypy.tools.json_in()
        def init(self):
            data = cherrypy.request.json
            assert 'width' in data, "Please provide parameter 'width'"
            assert 'height' in data, "Please provide parameter 'height'"
            Interface.initialized = True
            self.width = data['width']
            self.height = data['height']

        @cherrypy.expose
        @cherrypy.tools.json_in()
        def goal(self):
            data = cherrypy.request.json
            assert Interface.initialized, "Please call /init first!"
            assert 'agv' in data, "Please provide parameter 'agv'"
            assert 'x' in data, "Please provide parameter 'x'"
            assert 'y' in data, "Please provide parameter 'y'"
            assert 'theta' in data, "Please provide parameter 'theta'"
            print(data)
            self.goal = (data['x'], data['y'], data['theta'])
            self.agv = data['agv']


def map_to_plan(x_map):
    assert x_map.__class__ == tuple, "Coordinates should be tuples"
    return tuple(np.array(x_map) * scale + delta)


def plan_to_map(x_plan):
    assert x_plan.__class__ == tuple, "Coordinates should be tuples"
    return tuple((np.array(x_plan) - delta) / scale)



if __name__ == "__main__":
    cherrypy.config.update({'server.socket_port': 5432})
    cherrypy.quickstart(Interface())
    rospy.spin()
