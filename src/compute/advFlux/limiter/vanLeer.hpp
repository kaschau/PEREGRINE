// The van Leer limiter: smooth, and two at the limit.
#ifndef __limiterVanLeer_H__
#define __limiterVanLeer_H__

#include <Kokkos_Core.hpp>

struct vanLeer {
  static constexpr double limit = 2.0;
  static KOKKOS_INLINE_FUNCTION double phi(double r) {
    return (r + fabs(r)) / (1.0 + fabs(r));
  }
};

#endif
