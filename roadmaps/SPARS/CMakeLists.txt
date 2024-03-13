cmake_minimum_required(VERSION 3.5)
project(generateRoadmap)

# Enable C++14 and warnings
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_PREFIX_PATH /usr/local/lib)

# Set the python version
set(PYTHON_VERSION 310)
set(PYTHON_DOT_VERSION 3.10)
set(PYTHON_INCLUDE /usr/include/python${PYTHON_DOT_VERSION})
set(PYTHON_LIBRARY /usr/lib/python${PYTHON_DOT_VERSION}/config-${PYTHON_DOT_VERSION}m-x86_64-linux-gnu/)

FIND_PACKAGE(Boost 1.71.0 REQUIRED COMPONENTS 
    program_options 
    filesystem 
    serialization 
    python${PYTHON_VERSION}
)
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(yaml-cpp REQUIRED)
FIND_PACKAGE(ompl REQUIRED)
find_package( OpenCV REQUIRED )

# Additional include folders
include_directories(
  src
  ${EIGEN3_INCLUDE_DIR}
  ${OMPL_INCLUDE_DIRS}
  lodepng
  ${PYTHON_INCLUDE} 
  ${OpenCV_INCLUDE_DIRS}
)

# the executable
add_executable(generateRoadmap
  src/main.cpp
  lodepng/lodepng.cpp
)

target_link_libraries(generateRoadmap
  ${Boost_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ompl
)

# the python module
SET(SPARSPY sparspy)

add_library(${SPARSPY} SHARED 
  src/${SPARSPY}.cpp
  lodepng/lodepng.cpp
)

TARGET_LINK_LIBRARIES(
  ${SPARSPY} 
  ${Boost_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${OpenCV_LIBS}
  ompl
)

