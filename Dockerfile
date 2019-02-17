FROM ubuntu:bionic
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y python3 python3-pip python3-tk git iputils-ping

COPY roadmaps/adamsmap/requirements.txt /roadmaps/adamsmap/
WORKDIR /roadmaps/adamsmap
RUN pip3 install -r /roadmaps/adamsmap/requirements.txt

COPY roadmaps/adamsmap/* /roadmaps/adamsmap/
CMD ["python3", "adamsmap.py"]
