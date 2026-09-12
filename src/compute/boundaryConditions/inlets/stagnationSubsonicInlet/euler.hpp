#ifndef __stagnationSubsonicInlet_euler_H__
#define __stagnationSubsonicInlet_euler_H__

#include "boundaryConditions/faceRecords.hpp"

inline void stagnationSubsonicInlet_euler(const faceRecords &face, double tme) {
  auto q = face.q;
  auto qh = face.qh;
  auto S = face.S;
  auto qBcVals = face.qBcVals;
  const int nface = face.nface;
  const faceCells cells = faceCellsOf(face.d, nface);
  int firstHaloIdx = cells.halo, firstInteriorCellIdx = cells.interior,
      blockFaceIdx = cells.face, plus = cells.plus;

  auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
  auto qh1 = getFaceSlice(qh, nface, firstInteriorCellIdx);
  auto sVec = getFaceSlice(S, nface, blockFaceIdx);

  MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});

  for (int g = 0; g < ng; g++) {

    firstHaloIdx -= plus * g;
    auto q0 = getFaceSlice(q, nface, firstHaloIdx);

    Kokkos::parallel_for(
        "Constant velocity subsonic inlet euler terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          double S, nx, ny, nz;
          faceNormal(sVec(i, j, 0), sVec(i, j, 1), sVec(i, j, 2), S, nx, ny,
                     nz);

          // neumann total enthalpy, gamma to halo
          double &gamma = qh1(i, j, 0);
          double uxi = q1(i, j, 1) * nx;
          double uvi = q1(i, j, 2) * ny;
          double uwi = q1(i, j, 3) * nz;
          // Interior velo normal to face
          double Un = uxi + uvi + uwi;

          double V = sqrt(pow(q1(i, j, 1), 2.0) + pow(q1(i, j, 2), 2.0) +
                          pow(q1(i, j, 3), 2.0));
          double Ht =
              pow(qh1(i, j, 3), 2.0) / (gamma - 1.0) + 0.5 * pow(V, 2.0);
          double Jm = -Un + 2.0 * qh1(i, j, 3) / (gamma - 1.0);

          // solve quadratic for cb = -b/2a +/- sqrt(b**2-4ac)/2a
          double aq = 1 + 2.0 / (gamma - 1.0);
          double bq = -2.0 * Jm;
          double cq = (gamma - 1.0) * (0.5 * pow(Jm, 2.0) - Ht);
          double t1 = -bq / (2.0 * aq);
          double t2 = sqrt(pow(bq, 2.0) - 4.0 * aq * cq) / (2.0 * aq);

          double cb = fmax(t1 + t2, t1 - t2);

          // boundary velocity, Ma
          double Vb = 2.0 * cb / (gamma - 1.0) - Jm;
          double Mb = Vb / cb;

          // compute static pressure
          q0(i, j, 0) =
              qBcVals(i, j, 0) * pow(1.0 + (gamma - 1.0) / 2.0 * pow(Mb, 2.0),
                                     -gamma / (gamma - 1.0));

          // extrapolate face normal velocity
          q0(i, j, 1) = Vb * nx;
          q0(i, j, 2) = Vb * ny;
          q0(i, j, 3) = Vb * nz;

          // compute static temperature
          q0(i, j, 4) =
              qBcVals(i, j, 4) / (1.0 + (gamma - 1.0) / 2.0 * pow(Mb, 2.0));

          // apply species in halo
          for (int n = 5; n < ne; n++) {
            q0(i, j, n) = qBcVals(i, j, n);
          }
        });
  }
}

#endif
