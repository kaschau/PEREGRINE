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
#                             BUILD TYPES                                 #
###########################################################################
include( ./cmake/BUILD_TYPES.cmake )

###########################################################################
#                            COMPILER OPTIONS                             #
###########################################################################
include( ./cmake/COMPILER_OPTIONS.cmake )

###########################################################################
#                      RAPTOR COMPILE TIME OPTIONS                        #
###########################################################################
include( ./cmake/PEREGRINE_OPTIONS.cmake )

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
pybind11_add_module(Peregrine ${PROJECT_SOURCE_DIR}/Src/Peregrine.cpp)
target_include_directories(Peregrine PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(Peregrine PRIVATE peregrine_compute)
target_include_directories(Peregrine PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_compile_options(Peregrine PRIVATE -fvisibility=default)
