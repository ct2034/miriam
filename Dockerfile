FROM ubuntu:bionic
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y python3 python3-pip python3-tk cython3

COPY roadmaps/adamsmap/requirements.txt /roadmaps/adamsmap/
WORKDIR /roadmaps/adamsmap
RUN pip3 install -r /roadmaps/adamsmap/requirements.txt
RUN build.sh

COPY roadmaps/adamsmap/* /roadmaps/adamsmap/
CMD ["python3", "adamsmap_main.py", "dual.py"]
