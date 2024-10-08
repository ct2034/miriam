cmake_minimum_required(VERSION 3.5)
project(gsorm)

set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_PREFIX_PATH "$ENV{HOME}/bin/libtorch/share/cmake/Torch")

# Set the python version
set(PYTHON_VERSION 310)
set(PYTHON_DOT_VERSION 3.10)
set(PYTHON_INCLUDE /usr/include/python${PYTHON_DOT_VERSION})
set(PYTHON_LIBRARY /usr/lib/python${PYTHON_DOT_VERSION}/config-${PYTHON_DOT_VERSION}m-x86_64-linux-gnu/)

FIND_PACKAGE(Boost 1.71.0 REQUIRED COMPONENTS 
    program_options 
    python${PYTHON_VERSION}
)

# FIND_PACKAGE(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package( OpenCV REQUIRED )

include_directories(
  src
  lodepng
  ${OpenCV_INCLUDE_DIRS}
  ${PYTHON_INCLUDE}
)

# add_executable(${PROJECT_NAME}_main
#   src/gsorm.cpp
#   lodepng/lodepng.cpp
# )

# target_link_libraries(${PROJECT_NAME}_main
#     # "${TORCH_LIBRARIES}"
#     ${Boost_LIBRARIES}
#     ${OpenCV_LIBS}
# )

# the python module
SET(GSORM gsorm)

add_library(${GSORM} SHARED 
  src/${GSORM}.cpp
  lodepng/lodepng.cpp
)

TARGET_LINK_LIBRARIES(
  ${GSORM} 
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
)