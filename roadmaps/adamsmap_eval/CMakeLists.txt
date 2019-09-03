cmake_minimum_required(VERSION 2.8.3)
project(adamsmap_eval)

find_package(catkin REQUIRED COMPONENTS
  adamsmap
  rospy
)


catkin_package()
catkin_python_setup()


catkin_install_python(PROGRAMS
   script/eval_disc_main.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_add_nosetests(${CMAKE_CURRENT_SOURCE_DIR}/test)