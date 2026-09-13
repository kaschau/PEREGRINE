#ifndef __periodicRot_postDqDxyz_H__
#define __periodicRot_postDqDxyz_H__

#include "boundaryConditions/faceRecords.hpp"

inline void periodicRot_postDqDxyz(const faceRecords &face, double tme) {
  auto grads = face.grads;
  auto rot = face.rot;

  auto grads0 = face.halo(grads);

  MDRange4 range_face = face.range(1, ne);
  Kokkos::parallel_for(
      "Periodic postDqDxyz terms", range_face,
      KOKKOS_LAMBDA(const int g, const int i, const int j, const int l) {
        // turn the gradient vectors onto this face
        double grad[3] = {grads0(0, i, j, l, 0), grads0(0, i, j, l, 1),
                          grads0(0, i, j, l, 2)};
        for (int r = 0; r < 3; r++) {
          double turned = 0.0;
          for (int c = 0; c < 3; c++) {
            turned += rot(r, c) * grad[c];
          }
          grads0(0, i, j, l, r) = turned;
        }
      });
}

#endif
