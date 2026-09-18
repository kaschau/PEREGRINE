// The van Leer limiter: smooth, and two at the limit.
#ifndef __limiterVanLeer_H__
#define __limiterVanLeer_H__

#include "abi.hpp"

struct vanLeer {
  static constexpr fpdtype limit = 2.0;
  static KOKKOS_INLINE_FUNCTION fpdtype phi(fpdtype r) {
    return (r + fabs(r)) / (1.0 + fabs(r));
  }
};

#endif
