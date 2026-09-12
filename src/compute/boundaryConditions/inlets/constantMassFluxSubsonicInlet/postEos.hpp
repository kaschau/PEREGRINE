#ifndef __constantMassFluxSubsonicInlet_postEos_H__
#define __constantMassFluxSubsonicInlet_postEos_H__

#include "boundaryConditions/faceRecords.hpp"

inline void constantMassFluxSubsonicInlet_postEos(const faceRecords &face,
                                                  double tme) {
  auto q = face.q;
  auto Q = face.Q;
  auto QBcVals = face.QBcVals;
  const int nface = face.nface;
  const faceCells cells = faceCellsOf(face.d, nface);
  int firstHaloIdx = cells.halo, firstInteriorCellIdx = cells.interior,
      plus = cells.plus;
  int secondInteriorCellIdx = firstInteriorCellIdx + plus;

  // the eos has run on the halo from python: density is valid
  auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
  MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
  // set momentums, and velocities to match the desired mass flux
  // NOTE: We have to be careful with the indexing to accomodate fourth
  // order. In particular, we cannot just use firstInteriorCellIdx for all
  // the extrapolations so we have to make blockFaceIdx start with
  // firstInteriorCellIdx then increment

  // Reset first slice indicies, and make blockFaceIdx start at
  // firstInteriorCellIdx
  firstHaloIdx += plus * (ng - 1);
  secondInteriorCellIdx -= plus * (ng - 1);
  secondInteriorCellIdx -= plus;

  for (int g = 0; g < ng; g++) {
    firstHaloIdx -= plus * g;
    secondInteriorCellIdx += plus * g;

    auto q0 = getFaceSlice(q, nface, firstHaloIdx);
    auto q2 = getFaceSlice(q, nface, secondInteriorCellIdx);
    auto Q0 = getFaceSlice(Q, nface, firstHaloIdx);
    auto Q2 = getFaceSlice(Q, nface, secondInteriorCellIdx);

    Kokkos::parallel_for(
        "Constant mass flux subsonic inlet euler terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          // Target rhoU
          double &rhou = QBcVals(i, j, 1);
          double &rhov = QBcVals(i, j, 2);
          double &rhow = QBcVals(i, j, 3);

          // Set the velocities in the halo such that
          // 1/2(rho1+rho2)*1/2(u1+u2) evaluates to our desired rhou
          q0(i, j, 1) = 4.0 * rhou / (Q0(i, j, 0) + Q2(i, j, 0)) - q2(i, j, 1);
          q0(i, j, 2) = 4.0 * rhov / (Q0(i, j, 0) + Q2(i, j, 0)) - q2(i, j, 2);
          q0(i, j, 3) = 4.0 * rhow / (Q0(i, j, 0) + Q2(i, j, 0)) - q2(i, j, 3);

          // update momentum
          double &rho = Q0(i, j, 0);
          Q0(i, j, 1) = q0(i, j, 1) * rho;
          Q0(i, j, 2) = q0(i, j, 2) * rho;
          Q0(i, j, 3) = q0(i, j, 3) * rho;

          // we have created ke in halo, compute that and add it to
          // the existing rhoE, which is just internal energy at this
          // point
          double tke = 0.5 *
                       (pow(q0(i, j, 1), 2.0) + pow(q0(i, j, 2), 2.0) +
                        pow(q0(i, j, 3), 2.0)) *
                       rho;
          Q0(i, j, 4) += tke;
        });
  }
}

#endif
