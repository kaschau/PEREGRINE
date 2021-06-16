###########################################################################
#                   PEREGRINE COMPILE TIME OPTIONS                        #
###########################################################################

if( NOT KOKKOS_BACKEND )
  set(KOKKOS_BACKEND "Serial" CACHE STRING "Choose Kokkos Backend." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE KOKKOS_BACKEND PROPERTY STRINGS "Serial" "OpenMP" "Cuda-UVM" "Cuda" "HIP" "Default")
endif()

if( ${KOKKOS_BACKEND} STREQUAL "OpenMP" )
   remove_definitions( -DKOKKOS* )
   add_compile_definitions( KOKKOS_ENABLE_OPENMP )
elseif( ${KOKKOS_BACKEND} STREQUAL "Cuda-UVM" )
   remove_definitions( -DKOKKOS* )
   add_compile_definitions( KOKKOS_ENABLE_CUDA_UVM )
elseif( ${KOKKOS_BACKEND} STREQUAL "Cuda" )
   remove_definitions( -DKOKKOS* )
   add_compile_definitions( KOKKOS_ENABLE_CUDA )
elseif( ${KOKKOS_BACKEND} STREQUAL "HIP" )
   remove_definitions( -DKOKKOS* )
   add_compile_definitions( KOKKOS_ENABLE_HIP )
elseif( ${KOKKOS_BACKEND} STREQUAL "Serial" )
   remove_definitions( -DKOKKOS* )
   add_compile_definitions( KOKKOS_ENABLE_SERIAL )
else()
   remove_definitions( -DKOKKOS* )
   add_compile_definitions( KOKKOS_ENABLE_DEAULT )
endif()
