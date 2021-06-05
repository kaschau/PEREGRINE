#include "kokkos2peregrine.hpp"

// The struct that is sent to the Peregrine compute units. Holds all the data arrays
// for each block. Also converted into python class for modifying in the
// python wrapper
struct block {
  int nblki;
  int nx,ny,nz;

  threeDview x,y,z;
};

