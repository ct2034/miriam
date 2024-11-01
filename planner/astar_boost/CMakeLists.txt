cmake_minimum_required(VERSION 3.6)
PROJECT(hello_ext)

set(PYTHON_VERSION 310)
set(PYTHON_DOT_VERSION 3.10)
set(PYTHON_INCLUDE /usr/include/python${PYTHON_DOT_VERSION})
set(PYTHON_LIBRARY /usr/lib/python${PYTHON_DOT_VERSION}/config-${PYTHON_DOT_VERSION}m-x86_64-linux-gnu/)

# Find Boost
find_package(Boost REQUIRED COMPONENTS
    # python-py${PYTHON_VERSION}
    python${PYTHON_VERSION}
    graph
    random
    )
# find_package(Python${PYTHON_VERSION})

include_directories(${PYTHON_INCLUDE} ${Boost_INCLUDE_DIRS})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -lrt -O3")

SET(LIBASTARGRAPH astar_graph)

add_library(${LIBASTARGRAPH} SHARED ${LIBASTARGRAPH}.cpp)
TARGET_LINK_LIBRARIES(${LIBASTARGRAPH} ${Boost_LIBRARIES})
