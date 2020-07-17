#!/bin/bash

TEST_COMMANDS=(
    # planner
    #"mypy planner/policylearn" TODO: fiy typing
    "pycodestyle planner/policylearn --exclude=planner/policylearn/libMultiRobotPlanning"
    "pytest planner/policylearn --ignore=planner/policylearn/libMultiRobotPlanning"

    # sim
    "mypy sim/decentralized"
    "pycodestyle sim/decentralized"
    "pytest --cov=sim.decentralized sim/decentralized"

    # scenarios
    "mypy scenarios"
    "pycodestyle scenarios"
    "pytest --cov=scenarios scenarios"
)

# ------------------

RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

evaluate () {
    echo -e "$BLUE>>>>>>>>>> Running >$@<$NC"
    $@
    RCODE="$?"
    if [ $RCODE -ne 0 ]
    then
        echo -e "$BLUE<<<<<<<<<< Return code $RED>$RCODE<$BLUE of >$@<$NC"
    else
        echo -e "$BLUE<<<<<<<<<< Return code >$RCODE< of >$@<$NC"
    fi
    return $RCODE
}

source setup.sh

declare -a RCODES
for ((i = 0; i < ${#TEST_COMMANDS[@]}; i++))
do
    evaluate "${TEST_COMMANDS[$i]}"
    RCODES+=($?)
done

for RCODE in ${RCODES[@]}
do
    if [ $RCODE -ne 0 ]
    then
        exit $RCODE
    fi
done