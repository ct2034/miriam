#!/bin/bash
echo $(pwd)
source ./primally/bin/activate
ls -lh ./scenarios
python PRIMAL2/TestGenerator.py -a True
ls -lh ./scenarios
# python PRIMAL2/TestingEnv.py
python PRIMAL2/MultiProcessingTestDriver.py
