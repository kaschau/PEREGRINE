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
project(peregrine LANGUAGES C CXX)
#Set the install path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

#Add all da files
file( GLOB peregrine_sources ${PROJECT_SOURCE_DIR}/Src/*.cpp )
add_library(peregrine SHARED ${peregrine_sources})
target_include_directories(peregrine PUBLIC ${PROJECT_SOURCE_DIR}/include)

#Link to Kokkos
Find_Package(Kokkos REQUIRED)
target_link_libraries(peregrine PUBLIC Kokkos::kokkos)

#Make the bindings for python import
add_subdirectory(external/pybind11)
file( GLOB perepute_sources ${PROJECT_SOURCE_DIR}/perepute/*.cpp )
pybind11_add_module(perepute ${perepute_sources})
target_link_libraries(perepute PRIVATE peregrine)
target_include_directories(perepute PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_options(perepute PRIVATE -fvisibility=default)
