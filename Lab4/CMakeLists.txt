cmake_minimum_required(VERSION 3.30)
project(Lab4)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Lab4 main.cpp)

target_link_libraries(Lab4 ${OpenCV_LIBS})
