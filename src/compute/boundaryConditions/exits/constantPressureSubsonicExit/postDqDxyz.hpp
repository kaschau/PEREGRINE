#ifndef __constantPressureSubsonicExit_postDqDxyz_H__
#define __constantPressureSubsonicExit_postDqDxyz_H__

#include "boundaryConditions/faceRecords.hpp"

inline void constantPressureSubsonicExit_postDqDxyz(const faceRecords &face,
                                                    double tme) {
  auto grads = face.grads;
  const int nface = face.nface;
  const faceCells cells = faceCellsOf(face.d, nface);
  int firstHaloIdx = cells.halo, firstInteriorCellIdx = cells.interior;

  // Only gets applied to first halo slice
  auto grads1 = getFaceSlice(grads, nface, firstInteriorCellIdx);

  auto grads0 = getFaceSlice(grads, nface, firstHaloIdx);

  MDRange3 range_face =
      MDRange3({0, 0, 0}, {static_cast<long>(grads1.extent(0)),
                           static_cast<long>(grads1.extent(1)), ne});
  Kokkos::parallel_for(
      "Constant pressure subsonic exit postDqDxyz terms", range_face,
      KOKKOS_LAMBDA(const int i, const int j, const int l) {
        // neumann all gradients
        for (int d = 0; d < 3; d++) {
          grads0(i, j, l, d) = grads1(i, j, l, d);
        }
      });
}

#endif
