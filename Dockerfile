FROM ubuntu:bionic
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y python3 python3-pip python3-tk python3-pytest git 

# our python reqs
COPY planner/policylearn/requirements.txt /planner/policylearn/
WORKDIR /planner/policylearn
RUN pip3 install -r requirements.txt

# ecbs
RUN apt-get install -y cmake libboost-dev libboost-program-options-dev libboost-regex-dev libyaml-cpp-dev
RUN git clone https://github.com/ct2034/libMultiRobotPlanning.git /libMultiRobotPlanning
RUN mkdir /libMultiRobotPlanning/build
WORKDIR /libMultiRobotPlanning/build
RUN cmake ..
RUN make ecbs
ENV PYTHONPATH=$PYTHONPATH:/libMultiRobotPlanning
RUN pip3 install -r /libMultiRobotPlanning/requirements.txt

# running it
COPY planner/policylearn/* /planner/policylearn/
WORKDIR /planner/policylearn
CMD ["py.test-3", "-v"]
