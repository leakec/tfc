# Create target libraries
add_library(BF BF.cxx)

# Get necessary Python packages
find_package(Python COMPONENTS Interpreter NumPy)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})

# Get CUDA packages
find_package(CUDA)
if(CUDA_FOUND)
	target_compile_definitions(BF PUBLIC "HAS_CUDA")
	include_directories(${CUDA_INCLUDE_DIRS})
endif(CUDA_FOUND)	

# Set up public headers 
set_target_properties(BF PROPERTIES
	   				     PUBLIC_HEADER "BF.h"
						 CXX_STANDARD 17)
install(TARGETS BF)
