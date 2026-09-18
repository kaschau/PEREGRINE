// What a Riemann solver takes from either side of a cell face: one state,
// however the reconstruction made it. The momentum is carried beside the
// velocity so a solver that flows momentum reads what the cell holds.
#ifndef __faceState_H__
#define __faceState_H__

#include "abi.hpp"

struct faceState {
  fpdtype rho, u, v, w, rhou, rhov, rhow, p, E, c;
};

#endif
