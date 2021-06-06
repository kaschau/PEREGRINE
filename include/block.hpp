#ifndef __block_H__
#define __block_H__

#include "kokkos2peregrine.hpp"

// The struct that is sent to the Peregrine compute units. Holds all the data arrays
// for each block. Also converted into python class for modifying in the
// python wrapper
struct block {
  int nblki;
  int ni,nj,nk;

  threeDview x,y,z;
};

#endif
