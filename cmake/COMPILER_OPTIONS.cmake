###########################################################################
#                            COMPILER OPTIONS                             #
###########################################################################

#__________________________________________________________________________
#
#--------------------------------- INTEL ----------------------------------
#__________________________________________________________________________

if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel" )

# Release build type compiler flags
  set( CMAKE_CXX_FLAGS_RELEASE
       "-O2 \
       " CACHE STRING "" FORCE )

# Debug build type compiler flags
  set( CMAKE_CXX_FLAGS_DEBUG
       "-O0 \
        -g \
        -C \
        -traceback \
        -warn alignments \
        -warn general \
        -warn uncalled \
        -warn uninitialized \
        -warn usage \
        -check bounds \
        -check uninit \
        -check format" CACHE STRING "" FORCE )

endif()

#__________________________________________________________________________
#
#---------------------------------- GNU -----------------------------------
#__________________________________________________________________________

if( ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" )

# Dont allow VLAs so we catch them here
  set( CMAKE_CXX_FLAGS
       "-Werror=vla \
       " CACHE STRING "" FORCE )

# Release build type compiler flags
  set( CMAKE_CXX_FLAGS_RELEASE
       "-O2 \
       " CACHE STRING "" FORCE )

# Debug build type compiler flags
  set( CMAKE_CXX_FLAGS_DEBUG
       "-O0 \
        -g " CACHE STRING "" FORCE )

endif()
