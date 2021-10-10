FROM ubuntu:bionic
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y python3 python3-pip python3-tk python3-pytest git
# planner
COPY /planner/policylearn/requirements.txt /planner/policylearn/requirements.txt
RUN pip3 install -r /planner/policylearn/requirements.txt
COPY /planner/mapf_implementations/requirements.txt /planner/mapf_implementations/requirements.txt
RUN pip3 install -r /planner/mapf_implementations/requirements.txt
COPY /sim/decentralized/requirements.txt /sim/decentralized/requirements.txt
RUN pip3 install -r /sim/decentralized/requirements.txt
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN mkdir cache

# ecbs
RUN apt-get install -y cmake libboost-dev libboost-program-options-dev libboost-regex-dev libyaml-cpp-dev
COPY /planner/mapf_implementations/libMultiRobotPlanning /planner/mapf_implementations/libMultiRobotPlanning
RUN mkdir /planner/mapf_implementations/libMultiRobotPlanning/build
WORKDIR /planner/mapf_implementations/libMultiRobotPlanning/build
RUN cmake ..
RUN make ecbs

# doing it
ENV PYTHONPATH="/:/planner/mapf_implementations/libMultiRobotPlanning:${PYTHONPATH}"
COPY . /
WORKDIR /planner/policylearn
CMD [\
    "./generate_data.py", "no_solution" \
    ]
