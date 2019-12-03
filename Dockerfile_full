FROM ubuntu:xenial

# REQUIREMENTS
RUN apt-get update

# bonmin
RUN apt-get install -y libblas3 liblapack3 wget
#####

# roadmap
# RUN apt-get install -y sudo cmake libcgal-dev libcgal-qt5-dev libeigen3-dev libmgl-dev mathgl qt5-default libboost-filesystem-dev
# RUN wget http://ompl.kavrakilab.org/core/install-ompl-ubuntu.sh
# RUN sh install-ompl-ubuntu.sh
#####

# python
RUN apt-get install -y python3 python3-pip python3-tk git iputils-ping
COPY requirements.txt /
RUN mkdir /miriam/
RUN pip3 install -r requirements.txt
#####

# bonmin
RUN wget https://www.coin-or.org/download/binary/CoinAll/CoinAll-1.6.0-linux-x86_64-gcc4.4.5.tgz
RUN tar -xzf CoinAll-1.6.0-linux-x86_64-gcc4.4.5.tgz
RUN ln -s /usr/lib/liblapack.so.3 /lib/liblapack.so.3gf
RUN ln -s /usr/lib/libblas.so.3 /lib/libblas.so.3gf
#####

# cobra
RUN apt-get update
RUN apt-get -y install git gcc make libboost-all-dev g++

RUN git clone https://github.com/ct2034/cobra.git
WORKDIR cobra/COBRA
RUN make all
RUN cp cobra /miriam/
RUN chmod +x /miriam/cobra
ENV COBRA_BIN='/miriam/cobra'
#####

COPY . /miriam/

# roadmap
# RUN mkdir -p /miriam/roadmaps/prm-star/prm-star/build
# WORKDIR /miriam/roadmaps/prm-star/prm-star/build
# RUN cmake ..
# RUN make all
#####

# python
WORKDIR /miriam
ENV PYTHONPATH /miriam
CMD ["py.test","/miriam/.","-vs"]
#####
