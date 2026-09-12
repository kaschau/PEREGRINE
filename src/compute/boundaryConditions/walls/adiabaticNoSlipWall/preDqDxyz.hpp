#ifndef __adiabaticNoSlipWall_preDqDxyz_H__
#define __adiabaticNoSlipWall_preDqDxyz_H__

#include "boundaryConditions/faceRecords.hpp"

// Strategy for wall halo velocities:
//  For euler boundary conditions, we make all walls
//  slip walls. This is for computation of inviscid fluxes.
//  Then we apply the viscous bcs ("preDqDxyz")and make no
//  slip walls correct, so that velocity gradients will be correct
//  on no slip wall faces. After gradients ("postDqDxyz") we apply
//  the velocity gradients in the halos to have desired effect.
inline void adiabaticNoSlipWall_preDqDxyz(const faceRecords &face, double tme) {
  auto q = face.q;
  const int nface = face.nface;
  const faceCells cells = faceCellsOf(face.d, nface);
  int firstHaloIdx = cells.halo, firstInteriorCellIdx = cells.interior,
      plus = cells.plus;

  auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
  MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
  for (int g = 0; g < ng; g++) {
    firstHaloIdx -= plus * g;

    auto q0 = getFaceSlice(q, nface, firstHaloIdx);

    Kokkos::parallel_for(
        "Adia no slip wall preDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          // flip velo on wall
          q0(i, j, 1) = -q1(i, j, 1);
          q0(i, j, 2) = -q1(i, j, 2);
          q0(i, j, 3) = -q1(i, j, 3);
        });
  }
}

#endif
