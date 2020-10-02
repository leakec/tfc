cmake_minimum_required(VERSION 3.12)

project(TFC)

# Add extra cmake files
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/src/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -Wno-unused-parameter -fPIC -O3")

# Add src subdirectories for code
add_subdirectory(src)

