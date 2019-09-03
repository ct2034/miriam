cmake_minimum_required(VERSION 2.8.3)
project(adamsmap)

find_package(catkin REQUIRED)

catkin_python_setup()
catkin_package(
)

include_directories(${catkin_INCLUDE_DIRS})

# CYTHON MODULE
set(adamsmap_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src/adamsmap)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${adamsmap_ROOT}/cmake)
include(UseCython)
add_custom_target(ReplicatePythonSourceTree ALL ${CMAKE_COMMAND} -P
  ${adamsmap_ROOT}/cmake/ReplicatePythonSourceTree.cmake
  ${CMAKE_CURRENT_BINARY_DIR}
  WORKING_DIRECTORY ${adamsmap_ROOT})

set(adamsmap_ROS TRUE)
# set_source_files_properties(${adamsmap_ROOT}/adamsmap.pyx PROPERTIES CYTHON_IS_CXX FALSE)
add_subdirectory(${adamsmap_ROOT}/src)

install(PROGRAMS
  scripts/adamsmap
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
