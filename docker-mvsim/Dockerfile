FROM ros:kinetic-ros-base

RUN apt-get update 
RUN apt-get install -y ros-kinetic-mvsim
RUN apt-get install -y ros-kinetic-rviz
RUN apt-get install -y ros-kinetic-rviz-plugin-tutorials
RUN rm -rf /var/lib/apt/lists/
RUN touch ./root/bashrc && echo "source /opt/ros/kinetic/setup.bash" >> /root/.bashrc