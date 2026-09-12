#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <string.h>

PG_ABI void pgPeriodicRot(const pgView *q_, const pgView *Q_, const pgView *qh_,
                          const pgView *grads_, const pgView *S_,
                          const pgView *qBcVals_, const pgView *QBcVals_,
                          const pgView *rot_, const pgDims *d, int nface,
                          int terms, double tme) {
  auto q = as4(*q_), Q = as4(*Q_), qh = as4(*qh_);
  auto grads = as5(*grads_);
  auto S = as4(*S_);
  auto qBcVals = as3(*qBcVals_), QBcVals = as3(*QBcVals_);
  auto rot = as2(*rot_);
  const int ni = d->ni, nj = d->nj, nk = d->nk;
  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const faceCells f = faceCellsOf(*d, nface);
  int firstHaloIdx = f.halo, firstInteriorCellIdx = f.interior,
      blockFaceIdx = f.face, plus = f.plus;

  if (terms == 0) {

    auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < ng; g++) {
      firstHaloIdx -= plus * g;

      auto q0 = getFaceSlice(q, nface, firstHaloIdx);
      auto Q0 = getFaceSlice(Q, nface, firstHaloIdx);

      Kokkos::parallel_for(
          "Rotate periodic euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // turn the velocity vector onto this face
            double tempU, tempV, tempW;
            double u = q0(i, j, 1);
            double v = q0(i, j, 2);
            double w = q0(i, j, 3);
            tempU = rot(0, 0) * u + rot(0, 1) * v + rot(0, 2) * w;
            tempV = rot(1, 0) * u + rot(1, 1) * v + rot(1, 2) * w;
            tempW = rot(2, 0) * u + rot(2, 1) * v + rot(2, 2) * w;

            // Update velocity
            q0(i, j, 1) = tempU;
            q0(i, j, 2) = tempV;
            q0(i, j, 3) = tempW;

            // Update momentum
            Q0(i, j, 1) = tempU * Q0(i, j, 0);
            Q0(i, j, 2) = tempV * Q0(i, j, 0);
            Q0(i, j, 3) = tempW * Q0(i, j, 0);
          });
    }
  } else if (terms == 2) {

    auto grads0 = getFaceSlice(grads, nface, firstHaloIdx);

    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(grads0.extent(0)),
                             static_cast<long>(grads0.extent(1)), ne});
    Kokkos::parallel_for(
        "Periodic postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j, const int l) {
          // turn the gradient vectors onto this face
          double grad[3] = {grads0(i, j, l, 0), grads0(i, j, l, 1),
                            grads0(i, j, l, 2)};
          for (int r = 0; r < 3; r++) {
            double turned = 0.0;
            for (int c = 0; c < 3; c++) {
              turned += rot(r, c) * grad[c];
            }
            grads0(i, j, l, r) = turned;
          }
        });
  }
}
