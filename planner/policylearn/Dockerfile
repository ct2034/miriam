FROM tensorflow/tensorflow:nightly-gpu-py3

COPY . /policylearn
WORKDIR /policylearn
RUN pip install -r requirements.txt

CMD ["./train_model_classification.py", "training_data_class.pkl"]