#ifndef __supersonicExit_postDqDxyz_H__
#define __supersonicExit_postDqDxyz_H__

#include "boundaryConditions/faceRecords.hpp"

inline void supersonicExit_postDqDxyz(const faceRecords &face, double tme) {
  auto grads = face.grads;

  // Only applied to first halo slice
  auto grads1 = face.interior(grads);

  auto grads0 = face.halo(grads);

  MDRange4 range_face = face.range(1, ne);
  Kokkos::parallel_for(
      "Supersonic exit postDqDxyz terms", range_face,
      KOKKOS_LAMBDA(const int g, const int i, const int j, const int l) {
        // neumann all gradients
        for (int d = 0; d < 3; d++) {
          grads0(0, i, j, l, d) = grads1(0, i, j, l, d);
        }
      });
}

#endif
