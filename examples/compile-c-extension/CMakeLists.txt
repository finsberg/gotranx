cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(cellmodel LANGUAGES C)

set(CMAKE_POSITION_INDEPENDENT_CODE True)

if("${PROJECT_BINARY_DIR}" STREQUAL "${PROJECT_SOURCE_DIR}")
    message(FATAL_ERROR "You cannot build in the source directory. Please run cmake from a subdirectory called 'build'")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# add_executable(${EXECUTABLE_OUTPUT_PATH} demo.c)
add_library(${CELL_LIBFILE} SHARED demo.c)
install(TARGETS ${CELL_LIBFILE} DESTINATION lib)
