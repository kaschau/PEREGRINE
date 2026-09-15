###########################################################################
###########################################################################
##############   CMake Configuration for compute library  #################
###########################################################################
###########################################################################

###########################################################################
#                         CMake BOILER PLATE                              #
###########################################################################

cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
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

###########################################################################
#                            COMPILER OPTIONS                             #
###########################################################################
## INTEL SPECIFIC
if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=15009" CACHE STRING "" FORCE )
endif()
## GCC SPECIFIC
# if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" )
#
# endif()
## Add warnings
add_compile_options(-Wall)
###########################################################################
#              BUILD PEREGRINE COMPUTE LIBRARY                            #
###########################################################################
project(compute LANGUAGES C CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

#Turn off lto junk
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF )

# The runtime alone is built here: Kokkos and the memory the kernels run on.
# Every kernel is compiled on demand by peregrinepy.jit, with the compiler and
# flags this configure settled on.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_library(pgruntime SHARED ${PROJECT_SOURCE_DIR}/src/compute/runtime.cpp)
target_include_directories(pgruntime PUBLIC ${PROJECT_SOURCE_DIR}/src/compute)
set_target_properties(pgruntime PROPERTIES
                      CXX_VISIBILITY_PRESET hidden
                      VISIBILITY_INLINES_HIDDEN ON)

# every kernel library resolves Kokkos out of the runtime, so all of it is here
Find_Package(Kokkos REQUIRED)
target_link_libraries(pgruntime PUBLIC
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,Kokkos::kokkoscore,Kokkos::kokkoscontainers,Kokkos::kokkossimd>")
set_property(TARGET pgruntime PROPERTY CXX_STANDARD 20)

set(package_dir ${PROJECT_SOURCE_DIR}/src/peregrinepy)
install(TARGETS pgruntime LIBRARY DESTINATION ${package_dir})
# how runtime.cpp was compiled is how every kernel will be
install(CODE "execute_process(COMMAND ${Python3_EXECUTABLE} ${package_dir}/toolchain.py
                                      ${CMAKE_BINARY_DIR}/compile_commands.json
                                      ${CMAKE_SHARED_LIBRARY_SUFFIX}
                                      ${package_dir}/toolchain.json
                              COMMAND_ERROR_IS_FATAL ANY)")
