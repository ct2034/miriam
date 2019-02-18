FROM ubuntu:bionic
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y python3 python3-pip python3-tk git cython3

COPY roadmaps/adamsmap/requirements.txt /roadmaps/adamsmap/
WORKDIR /roadmaps/adamsmap
RUN pip3 install -r requirements.txt
RUN sh build.sh

COPY roadmaps/adamsmap/* /roadmaps/adamsmap/
RUN build.sh
CMD ["python3", "adamsmap_main.py", "dual.png"]
