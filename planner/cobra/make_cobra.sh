#/bin/bash
cd planner/cobra
while read p; do
  sudo apt install -y $p
done <apt_dependencies.txt

cd planner/cobra/external/COBRA
make
sudo ln -s $(pwd)/cobra /usr/bin/cobra
