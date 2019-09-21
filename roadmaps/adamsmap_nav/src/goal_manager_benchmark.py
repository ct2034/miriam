#!/usr/bin/env python
import pickle
from math import sqrt, pi

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseActionResult
from tf.transformations import quaternion_from_euler, euler_from_quaternion

SQRT2_2 = sqrt(2) / 2
PI_2 = pi / 2
GOAL_TOLERANCE_TRANS = .2
GOAL_TOLERANCE_ROT = .1


class GoalManager:
    def __init__(self, n_agents, poses):
        self.n_agents = n_agents
        self.pubs = []
        self.goals = []
        self.poses = poses
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.last_status = [0] * n_agents
        for i_a in range(n_agents):
            self.pubs.append(
                rospy.Publisher("/robot_{}/move_base_simple/goal".format(i_a), PoseStamped, queue_size=0)
            )
            self.goals.append(PoseStamped())
            self.goals[i_a].header.frame_id = 'map'
            rospy.Subscriber(
                name="/robot_{}/move_base/result".format(i_a),
                data_class=MoveBaseActionResult,
                callback=self.result_cb,
                callback_args=i_a
            )

    def retrieve_benchmark_metadata(self):
        global_planner = rospy.get_param("/robot_0/move_base/base_global_planner")
        graph_fname = rospy.get_param("/roadmap_server/graph_pkl")
        return global_planner.replace("/", "-"), graph_fname.split("/")[-1], "n_agents-" + str(self.n_agents)

    def set_goals_start(self):
        rospy.loginfo("Setting goals: start")
        self.set_goals("start")

    def set_goals_opposite(self):
        rospy.loginfo("Setting goals: opposite")
        self.set_goals("start")
        goals_copy = list(self.goals)
        for i_a in range(n_agents):
            self.goals[i_a] = goals_copy[(i_a + self.n_agents / 2) % self.n_agents]

    def set_goals(self, pose_name):
        these_poses = self.poses[pose_name]
        for i_a in range(n_agents):
            self.goals[i_a].pose.position.x = float(these_poses[str(i_a)][0])
            self.goals[i_a].pose.position.y = float(these_poses[str(i_a)][1])
            q = quaternion_from_euler(0, 0, float(these_poses[str(i_a)][2]))
            self.goals[i_a].pose.orientation.z = q[2]
            self.goals[i_a].pose.orientation.w = q[3]

    def publish_goals(self):
        rospy.loginfo("Publishing Goals ...")
        for i_a in range(n_agents):
            while self.pubs[i_a].get_num_connections() < 1:
                rospy.sleep(.1)
            self.pubs[i_a].publish(self.goals[i_a])

    def pub_and_wait(self):
        self.publish_goals()
        finished = [False] * n_agents
        i = 0
        start_t = rospy.Time.now()
        while not all(finished):
            i += 1
            if i % 100 == 0:
                self.publish_goals()
            rospy.sleep(.1)
            latest = rospy.Time(0)
            for i_a in range(n_agents):
                trans = self.tfBuffer.lookup_transform('map', 'robot_{}/base_link'.format(i_a), latest)
                dx = abs(trans.transform.translation.x - self.goals[i_a].pose.position.x)
                dy = abs(trans.transform.translation.y - self.goals[i_a].pose.position.y)
                goal_yaw = euler_from_quaternion([
                    0, 0,
                    self.goals[i_a].pose.orientation.z,
                    self.goals[i_a].pose.orientation.w])[2]
                current_yaw = euler_from_quaternion([
                    0, 0,
                    trans.transform.rotation.z,
                    trans.transform.rotation.w
                ])[2]
                dyaw = abs(goal_yaw - current_yaw)
                while dyaw > pi:
                    dyaw -= 2 * pi
                rospy.logdebug("i_a: {}, dx: {}, dy: {}, dyaw: {}".format(i_a, dx, dy, dyaw))
                if (
                        dx < GOAL_TOLERANCE_TRANS and
                        dy < GOAL_TOLERANCE_TRANS and
                        dyaw < GOAL_TOLERANCE_ROT and
                        self.last_status[i_a] == 3
                ):
                    finished[i_a] = True
        t = (rospy.Time.now() - start_t).to_sec()
        rospy.loginfo("All goals reached!\nTook {}s".format(t))
        return t

    def result_cb(self, msg, i_a):
        # msg = MoveBaseActionResult()
        self.last_status[i_a] = msg.status.status

    def save_data(self, data, folder):
        # type: (list, str)
        fname = folder + "_".join(self.retrieve_benchmark_metadata()) + ".pkl"
        with open(fname, "w") as f:
            pickle.dump(data, f)
        rospy.loginfo("saved data with len: {}".format(len(data)))
        rospy.loginfo(" .. to: {}".format(fname))


if __name__ == '__main__':
    rospy.init_node('goal_manager_benchmark', log_level=rospy.INFO)
    rospy.logdebug("init")

    n_agents = rospy.get_param("~n_agents")
    rospy.loginfo("n_agents: {}".format(n_agents))
    poses = rospy.get_param("poses")
    rospy.loginfo("available poses:\n{}".format("- \n".join(poses.keys())))
    benchmark_data_folder = rospy.get_param("~benchmark_data_folder")

    gm = GoalManager(n_agents, poses)
    ts = []
    gm.set_goals_start()  # preparation
    gm.pub_and_wait()

    for _ in range(25):
        gm.set_goals_opposite()
        t = gm.pub_and_wait()
        ts.append(t)
        gm.set_goals_start()
        t = gm.pub_and_wait()
        ts.append(t)
        gm.save_data(ts, benchmark_data_folder)
