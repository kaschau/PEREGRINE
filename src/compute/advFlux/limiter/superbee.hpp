// The superbee limiter: the most compressive that stays TVD.
#ifndef __limiterSuperbee_H__
#define __limiterSuperbee_H__

#include "abi.hpp"

struct superbee {
  static constexpr fpdtype limit = 2.0;
  static KOKKOS_INLINE_FUNCTION fpdtype phi(fpdtype r) {
    return fmax(0.0, fmax(fmin(2.0 * r, 1.0), fmin(r, 2.0)));
  }
};

#endif
