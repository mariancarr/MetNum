cmake_minimum_required(VERSION 3.18)
project(metodosnum)

set(CMAKE_CXX_STANDARD 14)

include_directories(${PROJECT_SOURCE_DIR}/external/eigen)
add_executable(metodosnum
        Tp2/main.cpp)
