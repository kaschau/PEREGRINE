#ifndef __constantMassFluxSubsonicInlet_euler_H__
#define __constantMassFluxSubsonicInlet_euler_H__

#include "boundaryConditions/faceRecords.hpp"

inline void constantMassFluxSubsonicInlet_euler(const faceRecords &face,
                                                double tme) {
  auto q = face.q;
  auto qBcVals = face.qBcVals;

  auto q1 = face.interior(q);
  MDRange3 range_face = face.range(ng);

  auto q0 = face.halo(q);

  Kokkos::parallel_for(
      "Constant mass flux subsonic inlet euler terms", range_face,
      KOKKOS_LAMBDA(const int g, const int i, const int j) {
        // extrapolate pressure
        q0(g, i, j, 0) = 2.0 * q1(0, i, j, 0) - q1(g + 1, i, j, 0);

        // apply zero velo to halo to make subsequent updates easier
        q0(g, i, j, 1) = 0.0;
        q0(g, i, j, 2) = 0.0;
        q0(g, i, j, 3) = 0.0;

        // apply temperature to halo
        q0(g, i, j, 4) = qBcVals(i, j, 4);

        // apply species to halo
        for (int n = 5; n < ne; n++) {
          q0(g, i, j, n) = qBcVals(i, j, n);
        }
      });
}

#endif
