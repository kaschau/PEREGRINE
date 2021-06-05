###########################################################################
###########################################################################
###############   CMake Configuration for libperegrine  ###################
###########################################################################
###########################################################################

###########################################################################
#                         CMake BOILER PLATE                              #
###########################################################################

cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
enable_language( C CXX )

###########################################################################
#                      BUILD PEREGRIME LIBRARY                            #
###########################################################################
project(peregrine_compute LANGUAGES C CXX)
#Set the install path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Lib)

#Add all da files
file( GLOB peregrine_compute_src ${PROJECT_SOURCE_DIR}/Src/compute/*.cpp )
add_library(peregrine_compute SHARED ${peregrine_compute_src})
target_include_directories(peregrine_compute PUBLIC ${PROJECT_SOURCE_DIR}/include)

#Link to Kokkos
Find_Package(Kokkos REQUIRED)
target_link_libraries(peregrine_compute PUBLIC Kokkos::kokkos)

#Make the bindings for python import
add_subdirectory(external/pybind11)
pybind11_add_module(peregrine ${PROJECT_SOURCE_DIR}/Src/peregrine.cpp)
target_include_directories(peregrine PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(peregrine PRIVATE peregrine_compute)
target_include_directories(peregrine PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_compile_options(peregrine PRIVATE -fvisibility=default)

