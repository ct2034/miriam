FROM ubuntu:bionic
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y python3 python3-pip python3-tk python3-pytest git cython3

COPY roadmaps/adamsmap/requirements.txt /roadmaps/adamsmap/
WORKDIR /roadmaps/adamsmap
RUN pip3 install -r requirements.txt

COPY roadmaps/adamsmap/* /roadmaps/adamsmap/
WORKDIR /roadmaps/adamsmap
RUN sh build.sh
RUN mkdir anim
CMD ["py.test-3", "-v"]
