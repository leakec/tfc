cmake_minimum_required(VERSION 3.12)

project(TFC)

# Doxygen documentation
find_package(Doxygen REQUIRED)

# Find all the public headers
file(GLOB_RECURSE PUBLIC_HEADERS ../src/cxx/*.h)
file(GLOB_RECURSE PYTHON_FILES ../src/python/*.py)

#This will be the main output of our command
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

#Replace variables inside @@ with the current values
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}) #Doxygen won't create this for us
add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
				   DEPENDS ${PUBLIC_HEADERS} ${PYTHON_FILES}
                   COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                   MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                   COMMENT "Generating docs"
				   VERBATIM)

add_custom_target(docs DEPENDS ${DOXYGEN_INDEX_FILE})
