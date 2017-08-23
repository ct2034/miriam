FROM python:latest

COPY requirements.txt /

RUN pip install -r requirements.txt

RUN mkdir /miriam/
COPY . /miriam/

CMD ["py.test","/miriam/.","-vs"]