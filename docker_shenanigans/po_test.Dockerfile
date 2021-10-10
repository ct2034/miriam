FROM ubuntu:bionic
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y python3 python3-pip python3-tk python3-pytest git
COPY . /

# planner
RUN pip3 install -r /planner/policylearn/requirements.txt
RUN pip3 install -r /planner/mapf_implementations/requirements.txt
RUN pip3 install -r /sim/decentralized/requirements.txt
RUN pip3 install -r requirements.txt
RUN mkdir cache

# ecbs
RUN apt-get install -y cmake libboost-dev libboost-program-options-dev libboost-regex-dev libyaml-cpp-dev
RUN mkdir /planner/mapf_implementations/libMultiRobotPlanning/build
WORKDIR /planner/mapf_implementations/libMultiRobotPlanning/build
RUN cmake ..
RUN make ecbs

# testing
WORKDIR /
CMD ["bash", "run_tests_policylearn.sh"]
