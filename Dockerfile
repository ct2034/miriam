FROM ubuntu:bionic
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y python3 python3-pip python3-tk python3-pytest git
COPY . /

# planner
# COPY planner /planner
RUN pip3 install -r /planner/policylearn/requirements.txt
RUN mkdir cache

# ecbs
RUN apt-get install -y cmake libboost-dev libboost-program-options-dev libboost-regex-dev libyaml-cpp-dev
RUN mkdir /planner/policylearn/libMultiRobotPlanning/build
WORKDIR /planner/policylearn/libMultiRobotPlanning/build
RUN cmake ..
RUN make ecbs
RUN pip3 install -r /planner/policylearn/libMultiRobotPlanning/requirements.txt

# sim decentralized
# COPY sim /sim
RUN pip3 install -r /sim/decentralized/requirements.txt

# testing
WORKDIR /
CMD ["bash", "run_tests_policylearn.sh"]
