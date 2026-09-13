#ifndef __constantPressureSubsonicExit_euler_H__
#define __constantPressureSubsonicExit_euler_H__

#include "boundaryConditions/faceRecords.hpp"

inline void constantPressureSubsonicExit_euler(const faceRecords &face,
                                               double tme) {
  auto q = face.q;
  auto S = face.S;
  auto qBcVals = face.qBcVals;

  auto q1 = face.interior(q);
  auto sVec = face.boundary(S);

  MDRange3 range_face = face.range(ng);
  const double dplus = face.outward();
  auto q0 = face.halo(q);

  Kokkos::parallel_for(
      "Constant pressure subsonic exit euler terms", range_face,
      KOKKOS_LAMBDA(const int g, const int i, const int j) {
        double S, nx, ny, nz;
        faceNormal(sVec(i, j, 0), sVec(i, j, 1), sVec(i, j, 2), S, nx, ny, nz);

        // set pressure
        q0(g, i, j, 0) = qBcVals(i, j, 0);

        // extrapolate velocity, unless reverse flow detected
        double uDotn =
            (q1(0, i, j, 1) * nx + q1(0, i, j, 2) * ny + q1(0, i, j, 3) * nz) *
            dplus;
        if (uDotn > 0.0) {
          for (int l = 1; l <= 3; l++) {
            q0(g, i, j, l) = 2.0 * q1(0, i, j, l) - q1(g + 1, i, j, l);
          }
        } else {
          // flip velocity on face (like slip wall)
          q0(g, i, j, 1) = q1(0, i, j, 1) - 2.0 * uDotn * nx * dplus;
          q0(g, i, j, 2) = q1(0, i, j, 2) - 2.0 * uDotn * ny * dplus;
          q0(g, i, j, 3) = q1(0, i, j, 3) - 2.0 * uDotn * nz * dplus;
        }

        // neumann everything else
        for (int l = 4; l < ne; l++) {
          q0(g, i, j, l) = q1(0, i, j, l);
        }
      });
}

#endif
