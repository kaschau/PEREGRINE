// What the dual time stages share: the Weiss and Smith preconditioning's
// reference velocity.
#ifndef __dualTime_H__
#define __dualTime_H__

#include "array"
#include "kernel.hpp"
#include "math.h"
#include "vector"
#include <Kokkos_Core.hpp>

//---------------------------------------------------------------------------------------------|
//
// Dual Time with preconditioning from
//
//     Preconditioning applied to variable and constant density flows
//     Weiss, Jonathan M. and Smith, Wayne A.
//     AIAA Journal
//     1995
//     doi: 10.2514/3.12946
//
//---------------------------------------------------------------------------------------------|

// Weiss & Smith reference velocity: the flow speed, clipped between eps*c and
// c, and no smaller than the diffusion velocity nu/dx. An inviscid call passes
// nu = 0 and a degenerate direction an infinite length, so neither binds.
static KOKKOS_INLINE_FUNCTION double
referenceVelocity(const double U, const double c, const double nu,
                  const double dI, const double dJ, const double dK) {
  const double eps = 1.0e-5;
  double Ur = fmax(U, eps * c);
  Ur = fmax(Ur, nu / dI);
  Ur = fmax(Ur, nu / dJ);
  Ur = fmax(Ur, nu / dK);
  return fmin(Ur, c);
}

#endif
