FROM ros:kinetic

RUN apt update
RUN apt install -y ros-kinetic-stage-ros \
  ros-kinetic-rosmon

COPY . /sim_stage

CMD ["ros_entryoint.sh", "roscore"]
