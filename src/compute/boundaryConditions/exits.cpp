#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <string.h>

PG_ABI void pgConstantPressureSubsonicExit(
    const pgView *q_, const pgView *Q_, const pgView *qh_, const pgView *grads_,
    const pgView *S_, const pgView *qBcVals_, const pgView *QBcVals_,
    const pgView *rot_, const pgDims *d, int nface, int terms, double tme) {
  auto q = as4(*q_), Q = as4(*Q_), qh = as4(*qh_);
  auto grads = as5(*grads_);
  auto S = as4(*S_);
  auto qBcVals = as3(*qBcVals_), QBcVals = as3(*QBcVals_);
  auto rot = as2(*rot_);
  const int ni = d->ni, nj = d->nj, nk = d->nk, ng = d->ng;
  const int ne = q.extent(3);
  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const faceCells f = faceCellsOf(*d, nface);
  int firstHaloIdx = f.halo, firstInteriorCellIdx = f.interior,
      blockFaceIdx = f.face, plus = f.plus;
  int secondInteriorCellIdx = firstInteriorCellIdx + plus;

  if (terms == 0) {

    auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
    auto sVec = getFaceSlice(S, nface, blockFaceIdx);

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    double dplus = -plus; // need outward normal
    for (int g = 0; g < ng; g++) {
      firstHaloIdx -= plus * g;
      auto q0 = getFaceSlice(q, nface, firstHaloIdx);
      secondInteriorCellIdx += plus * g;
      auto q2 = getFaceSlice(q, nface, secondInteriorCellIdx);

      Kokkos::parallel_for(
          "Constant pressure subsonic exit euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            double S, nx, ny, nz;
            faceNormal(sVec(i, j, 0), sVec(i, j, 1), sVec(i, j, 2), S, nx, ny,
                       nz);

            // set pressure
            q0(i, j, 0) = qBcVals(i, j, 0);

            // extrapolate velocity, unless reverse flow detected
            double uDotn =
                (q1(i, j, 1) * nx + q1(i, j, 2) * ny + q1(i, j, 3) * nz) *
                dplus;
            if (uDotn > 0.0) {
              for (int l = 1; l <= 3; l++) {
                q0(i, j, l) = 2.0 * q1(i, j, l) - q2(i, j, l);
              }
            } else {
              // flip velocity on face (like slip wall)
              q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx * dplus;
              q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny * dplus;
              q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz * dplus;
            }

            // neumann everything else
            for (int l = 4; l < ne; l++) {
              q0(i, j, l) = q1(i, j, l);
            }
          });
    }
  } else if (terms == 2) {

    // Only gets applied to first halo slice
    auto grads1 = getFaceSlice(grads, nface, firstInteriorCellIdx);

    auto grads0 = getFaceSlice(grads, nface, firstHaloIdx);

    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(grads1.extent(0)),
                             static_cast<long>(grads1.extent(1)), ne});
    Kokkos::parallel_for(
        "Constant pressure subsonic exit postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j, const int l) {
          // neumann all gradients
          for (int d = 0; d < 3; d++) {
            grads0(i, j, l, d) = grads1(i, j, l, d);
          }
        });
  }
}

PG_ABI void pgSupersonicExit(const pgView *q_, const pgView *Q_,
                             const pgView *qh_, const pgView *grads_,
                             const pgView *S_, const pgView *qBcVals_,
                             const pgView *QBcVals_, const pgView *rot_,
                             const pgDims *d, int nface, int terms,
                             double tme) {
  auto q = as4(*q_), Q = as4(*Q_), qh = as4(*qh_);
  auto grads = as5(*grads_);
  auto S = as4(*S_);
  auto qBcVals = as3(*qBcVals_), QBcVals = as3(*QBcVals_);
  auto rot = as2(*rot_);
  const int ni = d->ni, nj = d->nj, nk = d->nk, ng = d->ng;
  const int ne = q.extent(3);
  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const faceCells f = faceCellsOf(*d, nface);
  int firstHaloIdx = f.halo, firstInteriorCellIdx = f.interior,
      blockFaceIdx = f.face, plus = f.plus;
  int secondInteriorCellIdx = firstInteriorCellIdx + plus;

  if (terms == 0) {

    auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
    auto sVec = getFaceSlice(S, nface, blockFaceIdx);

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    double dplus = -plus; // need outward normal
    for (int g = 0; g < ng; g++) {
      firstHaloIdx -= plus * g;
      secondInteriorCellIdx += plus * g;

      auto q0 = getFaceSlice(q, nface, firstHaloIdx);
      auto q2 = getFaceSlice(q, nface, secondInteriorCellIdx);

      Kokkos::parallel_for(
          "Supersonic exit euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            double S, nx, ny, nz;
            faceNormal(sVec(i, j, 0), sVec(i, j, 1), sVec(i, j, 2), S, nx, ny,
                       nz);

            // extrapolate pressure (keep it positive, and wave exiting)
            q0(i, j, 0) =
                fmin(fmax(0.0, 2.0 * q1(i, j, 0) - q2(i, j, 0)), q1(i, j, 0));

            // extrapolate velocity, unless reverse flow detected
            double uDotn =
                (q1(i, j, 1) * nx + q1(i, j, 2) * ny + q1(i, j, 3) * nz) *
                dplus;
            if (uDotn > 0.0) {
              for (int l = 1; l <= 3; l++) {
                q0(i, j, l) = 2.0 * q1(i, j, l) - q2(i, j, l);
              }
            } else {
              // flip velocity on face (like slip wall)
              q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx * dplus;
              q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny * dplus;
              q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz * dplus;
            }

            // extrapolate temperature (keep it positive)
            q0(i, j, 4) = fmax(0.0, 2.0 * q1(i, j, 4) - q2(i, j, 4));
            // extrapolate species
            for (int l = 5; l < ne; l++) {
              q0(i, j, l) =
                  fmax(0.0, fmin(1.0, 2.0 * q1(i, j, l) - q2(i, j, l)));
            }
          });
    }
  } else if (terms == 2) {

    // Only applied to first halo slice
    auto grads1 = getFaceSlice(grads, nface, firstInteriorCellIdx);

    auto grads0 = getFaceSlice(grads, nface, firstHaloIdx);

    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(grads1.extent(0)),
                             static_cast<long>(grads1.extent(1)), ne});
    Kokkos::parallel_for(
        "Supersonic exit postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j, const int l) {
          // neumann all gradients
          for (int d = 0; d < 3; d++) {
            grads0(i, j, l, d) = grads1(i, j, l, d);
          }
        });
  }
}
