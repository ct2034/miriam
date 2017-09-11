#/bin/bash
git clone https://github.com/ct2034/cobra.git
cd cobra/COBRA
make
sudo ln -s $(pwd)/cobra /usr/bin/cobra
