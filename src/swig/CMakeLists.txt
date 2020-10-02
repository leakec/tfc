find_package(SWIG REQUIRED)

# Set policies to avoid warnings
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)

include(${SWIG_USE_FILE})
option(BUILD_PYTHON "Build Python SWIG module" ON)
if(BUILD_PYTHON)
    add_subdirectory(python)
endif()

