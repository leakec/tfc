set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Add cxx and swig subdirectories for C++ code
add_subdirectory(cxx)
add_subdirectory(swig)

# Add python directory for Python code
add_subdirectory(tfc)

# Add doc directory for code documentation
add_subdirectory(doc)

