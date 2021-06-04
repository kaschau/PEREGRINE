cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(peregrine LANGUAGES C CXX)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Lib)

add_library(peregrine SHARED user.cpp)
target_link_libraries(peregrine PUBLIC Kokkos::kokkos)

add_subdirectory(pybind11)

pybind11_add_module(ex_generate
    ${PROJECT_SOURCE_DIR}/ex-generate.cpp)

target_link_libraries(ex_generate PRIVATE peregrine)

## target_compile_options(ex_generate PRIVATE -fvisibility=default)
#configure_file(${PROJECT_SOURCE_DIR}/ex-numpy.py
#    ${CMAKE_BINARY_DIR}/ex-numpy.py @ONLY)
