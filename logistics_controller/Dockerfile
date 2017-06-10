FROM ros:kinetic-robot
# connect to host via bridge
ENV ROS_MASTER_URI=http://172.17.0.1:11311

RUN mkdir -p /ws/src/logistics_controller
COPY . /ws/src/logistics_controller
CMD bash -c "source /opt/ros/kinetic/setup.bash; \
            cd /ws/src; \
            catkin_init_workspace; \
            cd /ws; \
            catkin_make"

CMD bash -c "source /opt/ros/kinetic/setup.bash; roslaunch logistics_controller controller_2robots.launch"
