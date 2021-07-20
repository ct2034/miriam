cmake_minimum_required(VERSION 3.5)
project(generateRoadmap)

# Enable C++14 and warnings
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

FIND_PACKAGE(Boost 1.58 COMPONENTS program_options filesystem serialization REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
#FIND_PACKAGE(fcl REQUIRED)
FIND_PACKAGE(yaml-cpp REQUIRED)
# FIND_PACKAGE(assimp REQUIRED)
# FIND_PACKAGE(OMPL REQUIRED)
# FIND_PACKAGE(octomap REQUIRED)

# Additional include folders
include_directories(
  src
  ${EIGEN3_INCLUDE_DIR}
  ${OMPL_INCLUDE_DIRS}
  lodepng
  # ../libCommon/include
  # externalDependencies/ompl_bolt/src
)

add_executable(generateRoadmap
  src/main.cpp
  lodepng/lodepng.cpp
  # src/fclHelper.cpp
)

message(${OMPL_LIBRARIES})

target_link_libraries(generateRoadmap
  ${Boost_LIBRARIES}
  # ${fcl_LIBRARIES}
  # ${CCD_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  # ${ASSIMP_LIBRARIES}
  # ${OMPL_LIBRARIES}
  # ${OCTOMAP_LIBRARIES}
  # fcl
  # libCommon
  # ode
  # boost_serialization
  ompl
)

# target_compile_options(generateRoadmap PUBLIC -std=c++11 -Wall)
