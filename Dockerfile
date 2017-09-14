FROM python:3

COPY requirements.txt /

# python
RUN pip install -r requirements.txt

# bonmin
RUN apt-get update
RUN apt-get install -y libblas3 liblapack3 wget
RUN wget https://www.coin-or.org/download/binary/CoinAll/CoinAll-1.6.0-linux-x86_64-gcc4.4.5.tgz
RUN tar -xzf CoinAll-1.6.0-linux-x86_64-gcc4.4.5.tgz
RUN ln -s /usr/lib/liblapack.so.3 /lib/liblapack.so.3gf
RUN ln -s /usr/lib/libblas.so.3 /lib/libblas.so.3gf
###

# cobra
RUN chmod +x /miriam/cobra
COPY planner/cobra/libboost_graph.so.1.64.0 /usr/lib/
COPY planner/cobra/libboost_regex.so.1.64.0 /usr/lib/
###

RUN mkdir /miriam/
COPY . /miriam/
WORKDIR /miriam
ENV PYTHONPATH /miriam

CMD ["py.test","/miriam/.","-vs"]