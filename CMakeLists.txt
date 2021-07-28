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
#              BUILD PEREGRINE COMPUTE LIBRARY                            #
###########################################################################
project(compute LANGUAGES C CXX)
#Set the install path

#Turn off lto junk
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF )

#Add all da files
file( GLOB compute_src ${PROJECT_SOURCE_DIR}/src/compute/*.cpp
                       ${PROJECT_SOURCE_DIR}/src/compute/grid/*.cpp
                       ${PROJECT_SOURCE_DIR}/src/compute/flux/*.cpp
                       ${PROJECT_SOURCE_DIR}/src/compute/thermo/*.cpp
                       ${PROJECT_SOURCE_DIR}/src/compute/consistify/*.cpp)

include_directories(${PROJECT_SOURCE_DIR}/src/compute)


#Make the bindings for python import
add_subdirectory(lib/pybind11)
pybind11_add_module(compute ${compute_src} ${PROJECT_SOURCE_DIR}/src/bindings.cpp)

#Link to Kokkos
Find_Package(Kokkos REQUIRED)
target_link_libraries(compute PUBLIC Kokkos::kokkos)

message(STATUS "Kokkos Backend: " ${KOKKOS_BACKEND})
