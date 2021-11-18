#ifndef __face__H__
#define __face__H__

#include "kokkos_types.hpp"

// The struct that is sent to the Peregrine compute units. Holds all the data arrays
// for each block. Also converted into python class for modifying in the
// python wrapper
struct face_ {

  // face number
  int _nface;
  // Boundary condition value arrays
  threeDview qBcVals, QBcVals;

};

#endif
