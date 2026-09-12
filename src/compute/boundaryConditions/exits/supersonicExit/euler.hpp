#ifndef __supersonicExit_euler_H__
#define __supersonicExit_euler_H__

#include "boundaryConditions/faceRecords.hpp"

inline void supersonicExit_euler(const faceRecords &face, double tme) {
  auto q = face.q;
  auto S = face.S;
  const int nface = face.nface;
  const faceCells cells = faceCellsOf(face.d, nface);
  int firstHaloIdx = cells.halo, firstInteriorCellIdx = cells.interior,
      blockFaceIdx = cells.face, plus = cells.plus;
  int secondInteriorCellIdx = firstInteriorCellIdx + plus;

  auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
  auto sVec = getFaceSlice(S, nface, blockFaceIdx);

  MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
  double dplus = -plus; // need outward normal
  for (int g = 0; g < ng; g++) {
    firstHaloIdx -= plus * g;
    secondInteriorCellIdx += plus * g;

    auto q0 = getFaceSlice(q, nface, firstHaloIdx);
    auto q2 = getFaceSlice(q, nface, secondInteriorCellIdx);

    Kokkos::parallel_for(
        "Supersonic exit euler terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          double S, nx, ny, nz;
          faceNormal(sVec(i, j, 0), sVec(i, j, 1), sVec(i, j, 2), S, nx, ny,
                     nz);

          // extrapolate pressure (keep it positive, and wave exiting)
          q0(i, j, 0) =
              fmin(fmax(0.0, 2.0 * q1(i, j, 0) - q2(i, j, 0)), q1(i, j, 0));

          // extrapolate velocity, unless reverse flow detected
          double uDotn =
              (q1(i, j, 1) * nx + q1(i, j, 2) * ny + q1(i, j, 3) * nz) * dplus;
          if (uDotn > 0.0) {
            for (int l = 1; l <= 3; l++) {
              q0(i, j, l) = 2.0 * q1(i, j, l) - q2(i, j, l);
            }
          } else {
            // flip velocity on face (like slip wall)
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx * dplus;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny * dplus;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz * dplus;
          }

          // extrapolate temperature (keep it positive)
          q0(i, j, 4) = fmax(0.0, 2.0 * q1(i, j, 4) - q2(i, j, 4));
          // extrapolate species
          for (int l = 5; l < ne; l++) {
            q0(i, j, l) = fmax(0.0, fmin(1.0, 2.0 * q1(i, j, l) - q2(i, j, l)));
          }
        });
  }
}

#endif
