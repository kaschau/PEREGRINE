#ifndef __isoTNoSlipWall_preDqDxyz_H__
#define __isoTNoSlipWall_preDqDxyz_H__

#include "boundaryConditions/faceRecords.hpp"

// Strategy for wall halo velocities:
//  For euler boundary conditions, we make all walls
//  slip walls. This is for computation of inviscid fluxes.
//  Then we apply the viscous bcs ("preDqDxyz")and make no
//  slip walls correct, so that velocity gradients will be correct
//  on no slip wall faces. After gradients ("postDqDxyz") we apply
//  the velocity gradients in the halos to have desired effect.
inline void isoTNoSlipWall_preDqDxyz(const faceRecords &face, double tme) {
  auto q = face.q;

  auto q1 = face.interior(q);
  MDRange3 range_face = face.range(ng);
  auto q0 = face.halo(q);

  Kokkos::parallel_for(
      "isoT no slip wall preDqDxyz terms", range_face,
      KOKKOS_LAMBDA(const int g, const int i, const int j) {
        // flip velo on wall
        q0(g, i, j, 1) = -q1(0, i, j, 1);
        q0(g, i, j, 2) = -q1(0, i, j, 2);
        q0(g, i, j, 3) = -q1(0, i, j, 3);
      });
}

#endif
