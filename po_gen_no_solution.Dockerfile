FROM ubuntu:bionic
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y python3 python3-pip python3-tk python3-pytest git
# planner
COPY /planner/policylearn/requirements.txt /planner/policylearn/requirements.txt
RUN pip3 install -r /planner/policylearn/requirements.txt
COPY /planner/policylearn/libMultiRobotPlanning/requirements.txt /planner/policylearn/libMultiRobotPlanning/requirements.txt
RUN pip3 install -r /planner/policylearn/libMultiRobotPlanning/requirements.txt
COPY /sim/decentralized/requirements.txt /sim/decentralized/requirements.txt
RUN pip3 install -r /sim/decentralized/requirements.txt
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN mkdir cache

# ecbs
RUN apt-get install -y cmake libboost-dev libboost-program-options-dev libboost-regex-dev libyaml-cpp-dev
COPY /planner/policylearn/libMultiRobotPlanning /planner/policylearn/libMultiRobotPlanning
RUN mkdir /planner/policylearn/libMultiRobotPlanning/build
WORKDIR /planner/policylearn/libMultiRobotPlanning/build
RUN cmake ..
RUN make ecbs

# doing it
ENV PYTHONPATH="/:/planner/policylearn/libMultiRobotPlanning:${PYTHONPATH}"
COPY . /
WORKDIR /planner/policylearn
CMD [\
    "./generate_data.py", "no_solution" \
]
