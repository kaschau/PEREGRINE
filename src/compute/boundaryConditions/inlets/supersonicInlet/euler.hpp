#ifndef __supersonicInlet_euler_H__
#define __supersonicInlet_euler_H__

#include "boundaryConditions/faceRecords.hpp"

inline void supersonicInlet_euler(const faceRecords &face, double tme) {
  auto q = face.q;
  auto qBcVals = face.qBcVals;
  const int nface = face.nface;
  const faceCells cells = faceCellsOf(face.d, nface);
  int firstHaloIdx = cells.halo, firstInteriorCellIdx = cells.interior,
      plus = cells.plus;

  auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
  MDRange3 range_face =
      MDRange3({0, 0, 0}, {static_cast<int>(q1.extent(0)),
                           static_cast<int>(q1.extent(1)), ne});

  for (int g = 0; g < ng; g++) {
    firstHaloIdx -= plus * g;

    auto q0 = getFaceSlice(q, nface, firstHaloIdx);

    Kokkos::parallel_for(
        "Supersonic inlet euler terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j, const int l) {
          // apply all variables on face
          q0(i, j, l) = qBcVals(i, j, l);
        });
  }
}

#endif
