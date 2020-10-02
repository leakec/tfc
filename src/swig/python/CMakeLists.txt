# Include python and numpy directories from same python version
find_package(Python COMPONENTS Interpreter NumPy)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${Python_NumPy_INCLUDE_DIRS})

# Get CUDA packages
# Get CUDA packages
find_package(CUDA)
if(CUDA_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D HAS_CUDA")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D HAS_CUDA")
	include_directories(${CUDA_INCLUDE_DIRS})
	set_source_files_properties(../BF.i PROPERTIES COMPILE_DEFINITIONS HAS_CUDA)
endif(CUDA_FOUND)	

# Include BF library from previous cmake file
set(CMAKE_SWIG_FLAGS "-Wall")
set_source_files_properties(../BF.i PROPERTIES CPLUSPLUS ON)
include_directories(../../cxx)

# Add swig module
swig_add_library(BFSwig LANGUAGE python SOURCES ../BF.i)
swig_link_libraries(BFSwig BF ${PYTHON_LIBRARIES})

# Files to install with Python
set(PYTHON_INSTALL_FILES
	${CMAKE_CURRENT_BINARY_DIR}/BF.py
	${CMAKE_CURRENT_BINARY_DIR}/_BF.so)

# Configure setup.py and copy to output directory
set(SETUP_PY_IN ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in)
set(SETUP_PY_OUT ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
configure_file(${SETUP_PY_IN} ${SETUP_PY_OUT})

# Install target to call setup.py
add_custom_target(install-BF-python
		DEPENDS _BF
        COMMAND python ${SETUP_PY_OUT} install)
