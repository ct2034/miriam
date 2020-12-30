#/bin/bash
cd external/COBRA
make
sudo ln -s $(pwd)/cobra /usr/bin/cobra
