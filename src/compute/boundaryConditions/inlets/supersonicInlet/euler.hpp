#ifndef __supersonicInlet_euler_H__
#define __supersonicInlet_euler_H__

#include "boundaryConditions/faceRecords.hpp"

inline void supersonicInlet_euler(const faceRecords &face, double tme) {
  auto q = face.q;
  auto qBcVals = face.qBcVals;

  MDRange4 range_face = face.range(ng, ne);

  auto q0 = face.halo(q);

  Kokkos::parallel_for(
      "Supersonic inlet euler terms", range_face,
      KOKKOS_LAMBDA(const int g, const int i, const int j, const int l) {
        // apply all variables on face
        q0(g, i, j, l) = qBcVals(i, j, l);
      });
}

#endif
