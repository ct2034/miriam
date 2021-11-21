#!/bin/bash
echo pwd
source ./primally/bin/activate
python PRIMAL2/TestGenerator.py -a True
python PRIMAL2/TestingEnv.py