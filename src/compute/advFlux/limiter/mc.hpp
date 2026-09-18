// The monotonized central limiter: the generalized minmod at theta = 2,
// the least dissipative of that family.
#ifndef __limiterMc_H__
#define __limiterMc_H__

#include "abi.hpp"

struct mc {
  static constexpr fpdtype limit = 2.0;
  static KOKKOS_INLINE_FUNCTION fpdtype phi(fpdtype r) {
    return fmax(0.0, fmin(fmin(2.0 * r, (1.0 + r) / 2.0), 2.0));
  }
};

#endif
