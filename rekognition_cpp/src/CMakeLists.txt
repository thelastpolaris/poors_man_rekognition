# cmake version to be used
cmake_minimum_required( VERSION 3.0.2 )

# project name
project( PMR )
set(CMAKE_CXX_STANDARD 17)

set (PMR_VERSION_MAJOR 0)
set (PMR_VERSION_MINOR 0.01b)

# Get project directory
get_filename_component(BASE_PROJ_DIR ../ ABSOLUTE)

find_package( OpenCV REQUIRED )

# Includes
include_directories ("${PROJECT_SOURCE_DIR}/pipeline")
include_directories ("${PROJECT_SOURCE_DIR}/pipeline/pileine_elements")
include_directories ("${PROJECT_SOURCE_DIR}/pipeline/pipeline_elements/input_handlers")

# Source Files
aux_source_directory ("${PROJECT_SOURCE_DIR}" SOURCEFILE)
aux_source_directory ("${PROJECT_SOURCE_DIR}/pipeline/" SOURCEFILE)
aux_source_directory ("${PROJECT_SOURCE_DIR}/pipeline/pipeline_elements" SOURCEFILE)
aux_source_directory ("${PROJECT_SOURCE_DIR}/pipeline/pipeline_elements/input_handlers" SOURCEFILE)

# target
add_executable (PMR ${SOURCEFILE})
target_link_libraries( PMR ${OpenCV_LIBS} )
