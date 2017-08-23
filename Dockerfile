FROM python:latest

COPY . /

RUN pip install -r requirements.txt

CMD ["py.test" "-vs"]