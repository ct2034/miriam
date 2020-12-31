#/bin/bash
cd planner/cobra/external/COBRA
make
sudo ln -s $(pwd)/cobra /usr/bin/cobra
