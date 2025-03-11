cmake_minimum_required(VERSION 3.30)
project(Project)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Project main.cpp)

target_link_libraries(Project ${OpenCV_LIBS})