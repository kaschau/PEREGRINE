#ifndef __isoTNoSlipWall_euler_H__
#define __isoTNoSlipWall_euler_H__

#include "boundaryConditions/faceRecords.hpp"

// Strategy for wall halo velocities:
//  For euler boundary conditions, we make all walls
//  slip walls. This is for computation of inviscid fluxes.
//  Then we apply the viscous bcs ("preDqDxyz")and make no
//  slip walls correct, so that velocity gradients will be correct
//  on no slip wall faces. After gradients ("postDqDxyz") we apply
//  the velocity gradients in the halos to have desired effect.
inline void isoTNoSlipWall_euler(const faceRecords &face, double tme) {
  auto q = face.q;
  auto S = face.S;
  auto qBcVals = face.qBcVals;
  const int nface = face.nface;
  const faceCells cells = faceCellsOf(face.d, nface);
  int firstHaloIdx = cells.halo, firstInteriorCellIdx = cells.interior,
      blockFaceIdx = cells.face, plus = cells.plus;

  auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
  auto sVec = getFaceSlice(S, nface, blockFaceIdx);

  MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
  for (int g = 0; g < ng; g++) {
    firstHaloIdx -= plus * g;

    auto q0 = getFaceSlice(q, nface, firstHaloIdx);

    Kokkos::parallel_for(
        "isoT no slip wall euler terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          double S, nx, ny, nz;
          faceNormal(sVec(i, j, 0), sVec(i, j, 1), sVec(i, j, 2), S, nx, ny,
                     nz);

          // match pressure
          q0(i, j, 0) = q1(i, j, 0);

          // mirror velo on wall
          double uDotn = q1(i, j, 1) * nx + q1(i, j, 2) * ny + q1(i, j, 3) * nz;
          q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx;
          q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny;
          q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz;

          // set temperature
          q0(i, j, 4) = qBcVals(i, j, 4);
          // match species
          for (int n = 5; n < ne; n++) {
            q0(i, j, n) = q1(i, j, n);
          }
        });
  }
}

#endif
