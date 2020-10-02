cmake_minimum_required(VERSION 3.12)

project(TFC)

# Add extra cmake files
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -Wno-unused-parameter -fPIC -O3")

# For debugging only
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -Wall -Wextra -Wno-unused-parameter -fPIC")
#set(CMAKE_VERBOSE_MAKEFILE, TRUE)

set(PYTHON_SOURCE_FILES Latex.py;MakePlot.py;TFC.py;nTFC.py;TFCUtils.py;Html.py;DTFC.py)
foreach(CurrentPyFile IN LISTS PYTHON_SOURCE_FILES)
		configure_file(${PROJECT_SOURCE_DIR}/python/${CurrentPyFile} ${PROJECT_BINARY_DIR}/bin/${CurrentPyFile} COPYONLY)
endforeach()

# Add cxx and swig subdirectories for code
add_subdirectory(cxx)
add_subdirectory(swig)

# Add doc directory for code documentation
add_subdirectory(doc)

