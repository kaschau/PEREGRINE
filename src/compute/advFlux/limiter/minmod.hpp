// The minmod limiter: the smaller slope, or none where they disagree.
#ifndef __limiterMinmod_H__
#define __limiterMinmod_H__

#include "abi.hpp"

struct minmod {
  static constexpr fpdtype limit = 1.0;
  static KOKKOS_INLINE_FUNCTION fpdtype phi(fpdtype r) {
    return fmax(0.0, fmin(1.0, r));
  }
};

#endif
