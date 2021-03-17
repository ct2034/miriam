#!/bin/bash

for i in {00..09}
do
	echo "working on $i ..."
	./generate_data.py transfer_classification data/data_class$i.pkl data/data$i.pkl
done

