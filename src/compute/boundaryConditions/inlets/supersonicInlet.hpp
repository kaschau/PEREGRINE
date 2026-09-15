#ifndef __supersonicInlet_H__
#define __supersonicInlet_H__

#include "kernel.hpp"

namespace supersonicInlet {

struct euler {
  haloOut q;
  blockFaceIn qBcVals;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    for (int l = 0; l < ne; l++) {
      // apply all variables on face
      q.L(l) = qBcVals(l);
    }
  }
};

struct postDqDxyz {
  haloInOut grads;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    for (int l = 0; l < ne; l++) {
      // neumann all gradients
      for (int d = 0; d < 3; d++) {
        grads.L(l, d) = grads.R(l, d);
      }
    }
  }
};

} // namespace supersonicInlet

#endif
