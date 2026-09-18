#ifndef __supersonicInlet_H__
#define __supersonicInlet_H__

#include "boundaryConditions/haloState.hpp"
#include "kernel.hpp"

namespace supersonicInlet {

struct euler {
  haloInOut Q, q, qh;
  blockFaceIn qBcVals;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // the face's values, whole
    haloState(Q, q, qh, qBcVals(0), qBcVals(4), qBcVals(1), qBcVals(2),
              qBcVals(3), primitiveY(qBcVals));
  }
};

struct postDqDxyz {
  haloInOut grads;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // neumann all gradients
    for (int l = 0; l < ne - 1; l++) {
      for (int d = 0; d < 3; d++) {
        grads.L(l, d) = grads.R(l, d);
      }
    }
  }
};

} // namespace supersonicInlet

#endif
