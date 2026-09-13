#ifndef __constantVelocitySubsonicInlet_euler_H__
#define __constantVelocitySubsonicInlet_euler_H__

#include "boundaryConditions/faceRecords.hpp"

inline void constantVelocitySubsonicInlet_euler(const faceRecords &face,
                                                double tme) {
  auto q = face.q;
  auto qBcVals = face.qBcVals;

  auto q1 = face.interior(q);
  MDRange3 range_face = face.range(ng);

  auto q0 = face.halo(q);

  Kokkos::parallel_for(
      "Constant velocity subsonic inlet euler terms", range_face,
      KOKKOS_LAMBDA(const int g, const int i, const int j) {
        // extrapolate pressure
        q0(g, i, j, 0) = 2.0 * q1(0, i, j, 0) - q1(g + 1, i, j, 0);

        // apply velo in halo
        q0(g, i, j, 1) = qBcVals(i, j, 1);
        q0(g, i, j, 2) = qBcVals(i, j, 2);
        q0(g, i, j, 3) = qBcVals(i, j, 3);

        // apply temperature in halo
        q0(g, i, j, 4) = qBcVals(i, j, 4);

        // apply species in halo
        for (int n = 5; n < ne; n++) {
          q0(g, i, j, n) = qBcVals(i, j, n);
        }
      });
}

#endif
