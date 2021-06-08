#ifndef __Block_H__
#define __Block_H__

#include "kokkos_types.hpp"

// The struct that is sent to the Peregrine compute units. Holds all the data arrays
// for each block. Also converted into python class for modifying in the
// python wrapper
struct Block {
  int nblki;
  int ni,nj,nk;
  int ns=1;

  // Grid Arrays
  threeDview x,y,z;

  // Conserved Variables
  fourDview Qv;

  // Primatives
  threeDview T;
  threeDview p;
};

#endif
