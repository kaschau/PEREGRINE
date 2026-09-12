#ifndef __adiabaticSlipWall_postDqDxyz_H__
#define __adiabaticSlipWall_postDqDxyz_H__

#include "boundaryConditions/faceRecords.hpp"

// Strategy for wall halo velocities:
//  For euler boundary conditions, we make all walls
//  slip walls. This is for computation of inviscid fluxes.
//  Then we apply the viscous bcs ("preDqDxyz")and make no
//  slip walls correct, so that velocity gradients will be correct
//  on no slip wall faces. After gradients ("postDqDxyz") we apply
//  the velocity gradients in the halos to have desired effect.
inline void adiabaticSlipWall_postDqDxyz(const faceRecords &face, double tme) {
  auto grads = face.grads;
  const int nface = face.nface;
  const faceCells cells = faceCellsOf(face.d, nface);
  int firstHaloIdx = cells.halo, firstInteriorCellIdx = cells.interior;

  // Only applied to first halo slice
  auto grads0 = getFaceSlice(grads, nface, firstHaloIdx);

  auto grads1 = getFaceSlice(grads, nface, firstInteriorCellIdx);

  MDRange3 range_face =
      MDRange3({0, 0, 0}, {static_cast<long>(grads1.extent(0)),
                           static_cast<long>(grads1.extent(1)), ne});
  Kokkos::parallel_for(
      "Adia slip visc terms", range_face,
      KOKKOS_LAMBDA(const int i, const int j, const int l) {
        // negate all gradients
        for (int d = 0; d < 3; d++) {
          grads0(i, j, l, d) = -grads1(i, j, l, d);
        }
      });
}

#endif
