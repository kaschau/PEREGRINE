#ifndef __periodicRot_euler_H__
#define __periodicRot_euler_H__

#include "boundaryConditions/faceRecords.hpp"

inline void periodicRot_euler(const faceRecords &face, double tme) {
  auto q = face.q;
  auto Q = face.Q;
  auto rot = face.rot;

  MDRange3 range_face = face.range(ng);
  auto q0 = face.halo(q);
  auto Q0 = face.halo(Q);

  Kokkos::parallel_for(
      "Rotate periodic euler terms", range_face,
      KOKKOS_LAMBDA(const int g, const int i, const int j) {
        // turn the velocity vector onto this face
        double tempU, tempV, tempW;
        double u = q0(g, i, j, 1);
        double v = q0(g, i, j, 2);
        double w = q0(g, i, j, 3);
        tempU = rot(0, 0) * u + rot(0, 1) * v + rot(0, 2) * w;
        tempV = rot(1, 0) * u + rot(1, 1) * v + rot(1, 2) * w;
        tempW = rot(2, 0) * u + rot(2, 1) * v + rot(2, 2) * w;

        // Update velocity
        q0(g, i, j, 1) = tempU;
        q0(g, i, j, 2) = tempV;
        q0(g, i, j, 3) = tempW;

        // Update momentum
        Q0(g, i, j, 1) = tempU * Q0(g, i, j, 0);
        Q0(g, i, j, 2) = tempV * Q0(g, i, j, 0);
        Q0(g, i, j, 3) = tempW * Q0(g, i, j, 0);
      });
}

#endif
