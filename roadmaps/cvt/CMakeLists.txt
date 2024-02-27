cmake_minimum_required(VERSION 3.5)
project(cvt)

set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE "Release")

# Set the python version
set(PYTHON_VERSION 310)
set(PYTHON_DOT_VERSION 3.10)
set(PYTHON_INCLUDE /usr/include/python${PYTHON_DOT_VERSION})
set(PYTHON_LIBRARY /usr/lib/python${PYTHON_DOT_VERSION}/config-${PYTHON_DOT_VERSION}-x86_64-linux-gnu/libpython${PYTHON_DOT_VERSION}.so)

find_package(Boost 1.74.0 REQUIRED COMPONENTS 
    program_options 
    python${PYTHON_VERSION}
)

if(Boost_FOUND)
  message(STATUS "Boost found")
else()
  message(FATAL_ERROR "Boost not found")
endif()
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARY: ${Boost_LIBRARY}")
message(STATUS "Boost_LIBRARY_DIR: ${Boost_LIBRARY_DIR}")
# message(STATUS "Boost::program_options: ${Boost::program_options}")
# message(STATUS "Boost::python${PYTHON_VERSION}: ${Boost::python${PYTHON_VERSION}}")

find_package( OpenCV REQUIRED )
find_package( CGAL REQUIRED )

include_directories(
  src
  lodepng
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PYTHON_INCLUDE}
  ${CGAL_INCLUDE_DIRS}
)

# the python module
SET(cvt cvt)

add_library(${cvt} SHARED 
  src/${cvt}.cpp
  lodepng/lodepng.cpp
)

TARGET_LINK_LIBRARIES(
  ${cvt} 
  Boost::program_options
  Boost::python${PYTHON_VERSION}
  ${OpenCV_LIBS}
  ${PYTHON_LIBRARY}
  ${CGAL_LIBRARIES}
)