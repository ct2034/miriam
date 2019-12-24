FROM ubuntu:bionic
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y python3 python3-pip python3-tk python3-pytest git 

COPY planner/policylearn /planner/policylearn

# our python reqs
WORKDIR /planner/policylearn
RUN pip3 install -r requirements.txt

# ecbs
RUN apt-get install -y cmake libboost-dev libboost-program-options-dev libboost-regex-dev libyaml-cpp-dev
RUN mkdir /planner/policylearn/libMultiRobotPlanning/build
WORKDIR /planner/policylearn/libMultiRobotPlanning/build
RUN cmake ..
RUN make ecbs
RUN pip3 install -r /planner/policylearn/libMultiRobotPlanning/requirements.txt

# running it
WORKDIR /planner/policylearn
CMD ["py.test-3", "-v", "generate_data_test.py"]
