cmake_minimum_required(VERSION 3.25)

project(tfc)

# TODO: Change for release
add_compile_options(-Wall -Werror -pedantic)

# Contorl whether we build with shared libraries or static libraries
option(BUILD_SHARED_LIBS "Build using shared libraries" OFF)

# If not building with shared libs, set POSITION_INDEPENDENT_CODE 
# so that -fPIC gets used. This will
# allow linking the static libraries into the shared pybind11 libraries.
if(NOT BUILD_SHARED_LIBS)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()


# Use C++ 20
set(CMAKE_CXX_STANDARD 20)

# Turn on generation of compile_commands.json for LSP
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Turn on colored diagnostics
set(CMAKE_COLOR_DIAGNOSTICS ON)

# Find pybind11 in the system
# This is needed for CMake < 3.27. 
# After Cmake 3.27+, can remove setting PYBIND11_FINDPYTHON.
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 3.0 REQUIRED CONFIG)

# Create the bf library
add_library(bf BF.cc)
target_link_libraries(bf PUBLIC pybind11::pybind11)

# Create the BF.py Python file
pybind11_add_module(BF BF_Py.cc)
if (MINGW)
    target_link_libraries(BF PRIVATE bf -static -lpthread -static-libgcc -static-libstdc++)
else()
    target_link_libraries(BF PRIVATE bf)
endif()

install(TARGETS BF DESTINATION .)
install(CODE [=[execute_process(COMMAND stubgen -m BF -o . --include-docstring WORKING_DIRECTORY $ENV{DESTDIR}${CMAKE_INSTALL_PREFIX})]=])
