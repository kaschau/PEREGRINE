#ifndef __constantMassFluxSubsonicInlet_euler_H__
#define __constantMassFluxSubsonicInlet_euler_H__

#include "boundaryConditions/faceRecords.hpp"

inline void constantMassFluxSubsonicInlet_euler(const faceRecords &face,
                                                double tme) {
  auto q = face.q;
  auto qBcVals = face.qBcVals;
  const int nface = face.nface;
  const faceCells cells = faceCellsOf(face.d, nface);
  int firstHaloIdx = cells.halo, firstInteriorCellIdx = cells.interior,
      plus = cells.plus;
  int secondInteriorCellIdx = firstInteriorCellIdx + plus;

  auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
  MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});

  for (int g = 0; g < ng; g++) {
    firstHaloIdx -= plus * g;
    secondInteriorCellIdx += plus * g;

    auto q0 = getFaceSlice(q, nface, firstHaloIdx);
    auto q2 = getFaceSlice(q, nface, secondInteriorCellIdx);

    Kokkos::parallel_for(
        "Constant mass flux subsonic inlet euler terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          // extrapolate pressure
          q0(i, j, 0) = 2.0 * q1(i, j, 0) - q2(i, j, 0);

          // apply zero velo to halo to make subsequent updates easier
          q0(i, j, 1) = 0.0;
          q0(i, j, 2) = 0.0;
          q0(i, j, 3) = 0.0;

          // apply temperature to halo
          q0(i, j, 4) = qBcVals(i, j, 4);

          // apply species to halo
          for (int n = 5; n < ne; n++) {
            q0(i, j, n) = qBcVals(i, j, n);
          }
        });
  }
}

#endif
