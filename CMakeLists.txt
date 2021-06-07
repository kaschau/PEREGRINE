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
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Lib)

#Turn off lto junk
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF )

#Add all da files
file( GLOB compute_src ${PROJECT_SOURCE_DIR}/Src/compute/*.cpp )
add_library(compute SHARED ${compute_src})
target_include_directories(compute PUBLIC ${PROJECT_SOURCE_DIR}/include)

#Link to Kokkos
Find_Package(Kokkos REQUIRED)
target_link_libraries(compute PUBLIC Kokkos::kokkos)

#Make the bindings for python import
add_subdirectory(external/pybind11)
pybind11_add_module(Peregrine ${PROJECT_SOURCE_DIR}/Src/Peregrine.cpp)
target_include_directories(Peregrine PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(Peregrine PRIVATE compute)
target_include_directories(Peregrine PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_compile_options(Peregrine PRIVATE -fvisibility=default)
