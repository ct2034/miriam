FROM ros:kinetic-robot
# connect to host via bridge
ENV ROS_MASTER_URI=http://172.17.0.1:11311
ENV PACKAGE=multi_robot_nav

RUN apt-get update
RUN apt-get install -y\
 ros-kinetic-navigation 

SHELL ["bash", "-c"]
RUN mkdir -p /ws/src/$PACKAGE
COPY . /ws/src/$PACKAGE
RUN source /opt/ros/kinetic/setup.bash;\
 cd /ws/src;\
 catkin_init_workspace;\
 cd /ws;\
 catkin_make

CMD bash -c "/ros_entrypoint.sh;\
 export ROS_IP=$(hostname -I);\
 echo \"ip:$ROS_IP\";\
 source /ws/devel/setup.bash;\
 roslaunch $PACKAGE navigation.launch"
