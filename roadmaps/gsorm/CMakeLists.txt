cmake_minimum_required(VERSION 3.5)
project(gsorm)

set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_PREFIX_PATH "$ENV{HOME}/bin/libtorch/share/cmake/Torch")

FIND_PACKAGE(Boost 1.71.0 REQUIRED COMPONENTS 
    program_options 
)
FIND_PACKAGE(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package( OpenCV REQUIRED )

include_directories(
  src
  lodepng
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
  src/gsorm.cpp
  lodepng/lodepng.cpp
)

target_link_libraries(${PROJECT_NAME} 
    # "${TORCH_LIBRARIES}"
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
)
