cmake_minimum_required(VERSION 3.12)

project(TFC)

# Add extra cmake files
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/src/cmake" ${CMAKE_MODULE_PATH})

# Set CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -Wno-unused-parameter -fPIC -O3")

# Set default directory install
set(TFC_INSTALL_DIR "${CMAKE_SOURCE_DIR}/dist/tfc")

# Add src subdirectories for code
add_subdirectory(src)

