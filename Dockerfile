FROM ubuntu:bionic
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y python3 python3-pip python3-tk python3-pytest git

COPY planner/policylearn/requirements.txt /planner/policylearn/
WORKDIR /planner/policylearn
RUN pip3 install -r requirements.txt

COPY planner/policylearn/* /planner/policylearn/
WORKDIR /planner/policylearn
CMD ["py.test-3", "-v"]
