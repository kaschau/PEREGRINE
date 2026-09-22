// What a Riemann solver takes from either side of a cell face: one state,
// however the reconstruction made it. The momentum is what the cell holds;
// a solver that needs the velocity divides by the density.
#ifndef __faceState_H__
#define __faceState_H__

#include "abi.hpp"

struct faceState {
  fpdtype rho, rhou, rhov, rhow, p, E, c;
};

#endif
