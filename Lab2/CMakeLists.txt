cmake_minimum_required(VERSION 3.30)
project(Lab2)

set(CMAKE_CXX_STANDARD 23)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Lab2 main.cpp)

target_link_libraries(Lab2 ${OpenCV_LIBS})
