###########################################################################
###########################################################################
##############   CMake Configuration for compute library  #################
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
# Set a default build type if none was specified
set(default_build_type "RelWithDebInfo")
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(default_build_type "Debug")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option( PEREGRINE_NSCOMPILE "Compile compute with ns known" OFF )
if( PEREGRINE_NSCOMPILE )
   add_compile_definitions( NSCOMPILE )
   set( numSpecies "x" CACHE STRING "Number of species")
   add_compile_definitions(NS=${numSpecies})
else()
   remove_definitions( NS )
   unset( numSpecies CACHE)
endif()
###########################################################################
#                            COMPILER OPTIONS                             #
###########################################################################
## INTEL SPECIFIC
if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel" )
  set( CMAKE_CXX_FLAGS "-diag-disable=15009 -fp-model precise" CACHE STRING "" FORCE )
endif()
## GCC SPECIFIC
# if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" )
#
# endif()

###########################################################################
#              BUILD PEREGRINE COMPUTE LIBRARY                            #
###########################################################################
project(compute LANGUAGES C CXX)
#Set the install path

#Turn off lto junk
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF )

#Add all da files
file( GLOB compute_src ${PROJECT_SOURCE_DIR}/src/compute/*.cpp
                       ${PROJECT_SOURCE_DIR}/src/compute/advFlux/*.cpp
                       ${PROJECT_SOURCE_DIR}/src/compute/timeIntegration/*.cpp
                       ${PROJECT_SOURCE_DIR}/src/compute/boundaryConditions/*.cpp
                       ${PROJECT_SOURCE_DIR}/src/compute/diffFlux/*.cpp
                       ${PROJECT_SOURCE_DIR}/src/compute/thermo/*.cpp
                       ${PROJECT_SOURCE_DIR}/src/compute/switches/*.cpp
                       ${PROJECT_SOURCE_DIR}/src/compute/transport/*.cpp
                       ${PROJECT_SOURCE_DIR}/src/compute/chemistry/*.cpp
                       ${PROJECT_SOURCE_DIR}/src/compute/utils/*.cpp)

include_directories(${PROJECT_SOURCE_DIR}/src/compute)


#Make the bindings for python import
add_subdirectory(lib/pybind11)
pybind11_add_module(compute ${compute_src} ${PROJECT_SOURCE_DIR}/src/bindings.cpp)

#Link to Kokkos
Find_Package(Kokkos REQUIRED)
target_link_libraries(compute PUBLIC Kokkos::kokkos)

message(STATUS "Kokkos Backend: " ${KOKKOS_BACKEND})
