FROM ubuntu:bionic
RUN apt-get update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y python3 python3-pip python3-tk python3-pytest git cython3

COPY roadmaps/odrm/requirements.txt /roadmaps/odrm/
WORKDIR /roadmaps/odrm
RUN pip3 install -r requirements.txt

COPY roadmaps/odrm /roadmaps/odrm
WORKDIR /roadmaps/odrm
RUN python3 setup.py install
CMD ["py.test-3", "-v"]
