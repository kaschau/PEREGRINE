#ifndef __adiabaticMovingWall_postDqDxyz_H__
#define __adiabaticMovingWall_postDqDxyz_H__

#include "boundaryConditions/faceRecords.hpp"

// Strategy for wall halo velocities:
//  For euler boundary conditions, we make all walls
//  slip walls. This is for computation of inviscid fluxes.
//  Then we apply the viscous bcs ("preDqDxyz")and make no
//  slip walls correct, so that velocity gradients will be correct
//  on no slip wall faces. After gradients ("postDqDxyz") we apply
//  the velocity gradients in the halos to have desired effect.
inline void adiabaticMovingWall_postDqDxyz(const faceRecords &face,
                                           double tme) {
  auto grads = face.grads;

  auto grads0 = face.halo(grads);

  auto grads1 = face.interior(grads);

  MDRange3 range_face = face.range(1);
  Kokkos::parallel_for(
      "Adia moving wall postDqDxyz terms", range_face,
      KOKKOS_LAMBDA(const int g, const int i, const int j) {
        // negate pressure,  neumann velocity gradients
        grads0(0, i, j, 0, 0) = -grads1(0, i, j, 0, 0);
        grads0(0, i, j, 1, 0) = grads1(0, i, j, 1, 0);
        grads0(0, i, j, 2, 0) = grads1(0, i, j, 2, 0);
        grads0(0, i, j, 3, 0) = grads1(0, i, j, 3, 0);

        grads0(0, i, j, 0, 1) = -grads1(0, i, j, 0, 1);
        grads0(0, i, j, 1, 1) = grads1(0, i, j, 1, 1);
        grads0(0, i, j, 2, 1) = grads1(0, i, j, 2, 1);
        grads0(0, i, j, 3, 1) = grads1(0, i, j, 3, 1);

        grads0(0, i, j, 0, 2) = -grads1(0, i, j, 0, 2);
        grads0(0, i, j, 1, 2) = grads1(0, i, j, 1, 2);
        grads0(0, i, j, 2, 2) = grads1(0, i, j, 2, 2);
        grads0(0, i, j, 3, 2) = grads1(0, i, j, 3, 2);

        // negate temp and species gradient (so gradient evaluates to zero
        // on wall)
        grads0(0, i, j, 4, 0) = -grads1(0, i, j, 4, 0);
        grads0(0, i, j, 4, 1) = -grads1(0, i, j, 4, 1);
        grads0(0, i, j, 4, 2) = -grads1(0, i, j, 4, 2);

        for (int n = 5; n < ne; n++) {
          grads0(0, i, j, n, 0) = -grads1(0, i, j, n, 0);
          grads0(0, i, j, n, 1) = -grads1(0, i, j, n, 1);
          grads0(0, i, j, n, 2) = -grads1(0, i, j, n, 2);
        }
      });
}

#endif
