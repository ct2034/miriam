#!/usr/bin/env bash
docker run -it --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" $DOCKER_REG/ros:kinetic-mvsim roslaunch mvsim mvsim_demo_2robots.launch