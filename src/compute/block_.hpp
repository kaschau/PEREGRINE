#ifndef __block__H__
#define __block__H__

#include "kokkos_types.hpp"

// The struct that is sent to the Peregrine compute units. Holds all the data arrays
// for each block. Also converted into python class for modifying in the
// python wrapper
struct block_ {
  int nblki;
  int ni,nj,nk;
  int ns=1;

  // Grid Arrays
  threeDview x_,y_,z_;

  // Conserved Variables
  fourDview Q_;
  // Primative Variables
  threeDview q_;
};

#endif
