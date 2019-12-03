cmake_minimum_required(VERSION 2.8.3)
project(odrm)

find_package(catkin REQUIRED)

catkin_python_setup()
catkin_package(
)

include_directories(${catkin_INCLUDE_DIRS})

# CYTHON MODULE
set(odrm_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src/odrm)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${odrm_ROOT}/cmake)
include(UseCython)
add_custom_target(ReplicatePythonSourceTree ALL ${CMAKE_COMMAND} -P
  ${odrm_ROOT}/cmake/ReplicatePythonSourceTree.cmake
  ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${odrm_ROOT})

set(odrm_ROS TRUE)
# set_source_files_properties(${odrm_ROOT}/odrm.pyx PROPERTIES CYTHON_IS_CXX FALSE)
add_subdirectory(${odrm_ROOT}/src)

install(PROGRAMS
  scripts/odrm
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

catkin_add_nosetests(${CMAKE_CURRENT_SOURCE_DIR}/test)
