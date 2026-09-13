#ifndef __constantMassFluxSubsonicInlet_postEos_H__
#define __constantMassFluxSubsonicInlet_postEos_H__

#include "boundaryConditions/faceRecords.hpp"

inline void constantMassFluxSubsonicInlet_postEos(const faceRecords &face,
                                                  double tme) {
  auto QBcVals = face.QBcVals;
  auto qH = face.halo(face.q), qI = face.interior(face.q);
  auto QH = face.halo(face.Q), QI = face.interior(face.Q);

  // the eos has run on the halo from python: density is valid. The
  // velocities either side of the face are set so that
  // 1/2(rho1+rho2)*1/2(u1+u2) evaluates to the desired mass flux: each
  // interior layer but the outermost from the halo layer facing it, then the
  // first halo layer from the first interior one
  Kokkos::parallel_for(
      "Constant mass flux subsonic inlet postEos terms", face.range(ng),
      KOKKOS_LAMBDA(const int g, const int i, const int j) {
        const bool inward = g < ng - 1;
        const int k = inward ? ng - 2 - g : 0;
        const planes<double, 4> &q0 = inward ? qI : qH;
        const planes<double, 4> &Q0 = inward ? QI : QH;
        const planes<double, 4> &q2 = inward ? qH : qI;
        const planes<double, 4> &Q2 = inward ? QH : QI;

        // Target rhoU
        const double &rhou = QBcVals(i, j, 1);
        const double &rhov = QBcVals(i, j, 2);
        const double &rhow = QBcVals(i, j, 3);

        q0(k, i, j, 1) =
            4.0 * rhou / (Q0(k, i, j, 0) + Q2(k, i, j, 0)) - q2(k, i, j, 1);
        q0(k, i, j, 2) =
            4.0 * rhov / (Q0(k, i, j, 0) + Q2(k, i, j, 0)) - q2(k, i, j, 2);
        q0(k, i, j, 3) =
            4.0 * rhow / (Q0(k, i, j, 0) + Q2(k, i, j, 0)) - q2(k, i, j, 3);

        // update momentum
        double &rho = Q0(k, i, j, 0);
        Q0(k, i, j, 1) = q0(k, i, j, 1) * rho;
        Q0(k, i, j, 2) = q0(k, i, j, 2) * rho;
        Q0(k, i, j, 3) = q0(k, i, j, 3) * rho;
      });
}

#endif
