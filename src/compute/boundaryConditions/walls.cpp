#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <string.h>

// Strategy for wall halo velocities:
//  For euler boundary conditions, we make all walls
//  slip walls. This is for computation of inviscid fluxes.
//  Then we apply the viscous bcs ("preDqDxyz")and make no
//  slip walls correct, so that velocity gradients will be correct
//  on no slip wall faces. After gradients ("postDqDxyz") we apply
//  the velocity gradients in the halos to have desired effect.

PG_ABI void pgAdiabaticNoSlipWall(const pgView *q_, const pgView *Q_,
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

  if (terms == 0) {

    auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
    auto sVec = getFaceSlice(S, nface, blockFaceIdx);

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < ng; g++) {
      firstHaloIdx -= plus * g;
      auto q0 = getFaceSlice(q, nface, firstHaloIdx);

      Kokkos::parallel_for(
          "Adia no slip wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            double S, nx, ny, nz;
            faceNormal(sVec(i, j, 0), sVec(i, j, 1), sVec(i, j, 2), S, nx, ny,
                       nz);

            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn =
                q1(i, j, 1) * nx + q1(i, j, 2) * ny + q1(i, j, 3) * nz;
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz;

            // match temperature
            q0(i, j, 4) = q1(i, j, 4);
            // match species
            for (int n = 5; n < ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }

  } else if (terms == 1) {

    auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < ng; g++) {
      firstHaloIdx -= plus * g;

      auto q0 = getFaceSlice(q, nface, firstHaloIdx);

      Kokkos::parallel_for(
          "Adia no slip wall preDqDxyz terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // flip velo on wall
            q0(i, j, 1) = -q1(i, j, 1);
            q0(i, j, 2) = -q1(i, j, 2);
            q0(i, j, 3) = -q1(i, j, 3);
          });
    }
  } else if (terms == 2) {

    // Only applied to first halo slice
    auto grads0 = getFaceSlice(grads, nface, firstHaloIdx);

    auto grads1 = getFaceSlice(grads, nface, firstInteriorCellIdx);

    MDRange2 range_face =
        MDRange2({0, 0}, {grads1.extent(0), grads1.extent(1)});
    Kokkos::parallel_for(
        "Adia no slip postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          // negate pressure,  neumann velocity gradients
          grads0(i, j, 0, 0) = -grads1(i, j, 0, 0);
          grads0(i, j, 1, 0) = grads1(i, j, 1, 0);
          grads0(i, j, 2, 0) = grads1(i, j, 2, 0);
          grads0(i, j, 3, 0) = grads1(i, j, 3, 0);

          grads0(i, j, 0, 1) = -grads1(i, j, 0, 1);
          grads0(i, j, 1, 1) = grads1(i, j, 1, 1);
          grads0(i, j, 2, 1) = grads1(i, j, 2, 1);
          grads0(i, j, 3, 1) = grads1(i, j, 3, 1);

          grads0(i, j, 0, 2) = -grads1(i, j, 0, 2);
          grads0(i, j, 1, 2) = grads1(i, j, 1, 2);
          grads0(i, j, 2, 2) = grads1(i, j, 2, 2);
          grads0(i, j, 3, 2) = grads1(i, j, 3, 2);

          // negate temp and species gradient (so gradient evaluates to zero
          // on wall)
          grads0(i, j, 4, 0) = -grads1(i, j, 4, 0);
          grads0(i, j, 4, 1) = -grads1(i, j, 4, 1);
          grads0(i, j, 4, 2) = -grads1(i, j, 4, 2);

          for (int n = 5; n < ne; n++) {
            grads0(i, j, n, 0) = -grads1(i, j, n, 0);
            grads0(i, j, n, 1) = -grads1(i, j, n, 1);
            grads0(i, j, n, 2) = -grads1(i, j, n, 2);
          }
        });
  }
}

PG_ABI void pgAdiabaticSlipWall(const pgView *q_, const pgView *Q_,
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

  if (terms == 0) {

    auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
    auto sVec = getFaceSlice(S, nface, blockFaceIdx);

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < ng; g++) {
      firstHaloIdx -= plus * g;

      auto q0 = getFaceSlice(q, nface, firstHaloIdx);

      Kokkos::parallel_for(
          "Adia slip wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            double S, nx, ny, nz;
            faceNormal(sVec(i, j, 0), sVec(i, j, 1), sVec(i, j, 2), S, nx, ny,
                       nz);

            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn =
                q1(i, j, 1) * nx + q1(i, j, 2) * ny + q1(i, j, 3) * nz;
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz;

            // match temperature
            q0(i, j, 4) = q1(i, j, 4);
            // match species
            for (int n = 5; n < ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }
  } else if (terms == 2) {

    // Only applied to first halo slice
    auto grads0 = getFaceSlice(grads, nface, firstHaloIdx);

    auto grads1 = getFaceSlice(grads, nface, firstInteriorCellIdx);

    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(grads1.extent(0)),
                             static_cast<long>(grads1.extent(1)), ne});
    Kokkos::parallel_for(
        "Adia slip visc terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j, const int l) {
          // negate all gradients
          for (int d = 0; d < 3; d++) {
            grads0(i, j, l, d) = -grads1(i, j, l, d);
          }
        });
  }
}

PG_ABI void pgAdiabaticMovingWall(const pgView *q_, const pgView *Q_,
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

  if (terms == 0) {

    auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
    auto sVec = getFaceSlice(S, nface, blockFaceIdx);

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < ng; g++) {
      firstHaloIdx -= plus * g;

      auto q0 = getFaceSlice(q, nface, firstHaloIdx);

      Kokkos::parallel_for(
          "Adia moving wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            double S, nx, ny, nz;
            faceNormal(sVec(i, j, 0), sVec(i, j, 1), sVec(i, j, 2), S, nx, ny,
                       nz);

            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn =
                q1(i, j, 1) * nx + q1(i, j, 2) * ny + q1(i, j, 3) * nz;
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz;

            // match temperature
            q0(i, j, 4) = q1(i, j, 4);
            // match species
            for (int n = 5; n < ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }

  } else if (terms == 1) {

    auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < ng; g++) {
      firstHaloIdx -= plus * g;

      auto q0 = getFaceSlice(q, nface, firstHaloIdx);
      Kokkos::parallel_for(
          "Adia moving wall preDqDxyz terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // apply velo to face
            q0(i, j, 1) = 2.0 * qBcVals(i, j, 1) - q1(i, j, 1);
            q0(i, j, 2) = 2.0 * qBcVals(i, j, 2) - q1(i, j, 2);
            q0(i, j, 3) = 2.0 * qBcVals(i, j, 3) - q1(i, j, 3);
          });
    }
  } else if (terms == 2) {

    auto grads0 = getFaceSlice(grads, nface, firstHaloIdx);

    auto grads1 = getFaceSlice(grads, nface, firstInteriorCellIdx);

    MDRange2 range_face =
        MDRange2({0, 0}, {grads1.extent(0), grads1.extent(1)});
    Kokkos::parallel_for(
        "Adia moving wall postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          // negate pressure,  neumann velocity gradients
          grads0(i, j, 0, 0) = -grads1(i, j, 0, 0);
          grads0(i, j, 1, 0) = grads1(i, j, 1, 0);
          grads0(i, j, 2, 0) = grads1(i, j, 2, 0);
          grads0(i, j, 3, 0) = grads1(i, j, 3, 0);

          grads0(i, j, 0, 1) = -grads1(i, j, 0, 1);
          grads0(i, j, 1, 1) = grads1(i, j, 1, 1);
          grads0(i, j, 2, 1) = grads1(i, j, 2, 1);
          grads0(i, j, 3, 1) = grads1(i, j, 3, 1);

          grads0(i, j, 0, 2) = -grads1(i, j, 0, 2);
          grads0(i, j, 1, 2) = grads1(i, j, 1, 2);
          grads0(i, j, 2, 2) = grads1(i, j, 2, 2);
          grads0(i, j, 3, 2) = grads1(i, j, 3, 2);

          // negate temp and species gradient (so gradient evaluates to zero
          // on wall)
          grads0(i, j, 4, 0) = -grads1(i, j, 4, 0);
          grads0(i, j, 4, 1) = -grads1(i, j, 4, 1);
          grads0(i, j, 4, 2) = -grads1(i, j, 4, 2);

          for (int n = 5; n < ne; n++) {
            grads0(i, j, n, 0) = -grads1(i, j, n, 0);
            grads0(i, j, n, 1) = -grads1(i, j, n, 1);
            grads0(i, j, n, 2) = -grads1(i, j, n, 2);
          }
        });
  }
}

PG_ABI void pgIsoTNoSlipWall(const pgView *q_, const pgView *Q_,
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

  if (terms == 0) {

    auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
    auto sVec = getFaceSlice(S, nface, blockFaceIdx);

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < ng; g++) {
      firstHaloIdx -= plus * g;

      auto q0 = getFaceSlice(q, nface, firstHaloIdx);

      Kokkos::parallel_for(
          "isoT no slip wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            double S, nx, ny, nz;
            faceNormal(sVec(i, j, 0), sVec(i, j, 1), sVec(i, j, 2), S, nx, ny,
                       nz);

            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn =
                q1(i, j, 1) * nx + q1(i, j, 2) * ny + q1(i, j, 3) * nz;
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz;

            // set temperature
            q0(i, j, 4) = qBcVals(i, j, 4);
            // match species
            for (int n = 5; n < ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }

  } else if (terms == 1) {

    auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < ng; g++) {
      firstHaloIdx -= plus * g;

      auto q0 = getFaceSlice(q, nface, firstHaloIdx);

      Kokkos::parallel_for(
          "isoT no slip wall preDqDxyz terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // flip velo on wall
            q0(i, j, 1) = -q1(i, j, 1);
            q0(i, j, 2) = -q1(i, j, 2);
            q0(i, j, 3) = -q1(i, j, 3);
          });
    }
  } else if (terms == 2) {

    auto grads0 = getFaceSlice(grads, nface, firstHaloIdx);

    auto grads1 = getFaceSlice(grads, nface, firstInteriorCellIdx);

    MDRange2 range_face =
        MDRange2({0, 0}, {grads1.extent(0), grads1.extent(1)});
    Kokkos::parallel_for(
        "isoT no slip postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          // negate pressure,  neumann velocity,temperature gradients
          grads0(i, j, 0, 0) = -grads1(i, j, 0, 0);
          grads0(i, j, 1, 0) = grads1(i, j, 1, 0);
          grads0(i, j, 2, 0) = grads1(i, j, 2, 0);
          grads0(i, j, 3, 0) = grads1(i, j, 3, 0);
          grads0(i, j, 4, 0) = grads1(i, j, 4, 0);

          grads0(i, j, 0, 1) = -grads1(i, j, 0, 1);
          grads0(i, j, 1, 1) = grads1(i, j, 1, 1);
          grads0(i, j, 2, 1) = grads1(i, j, 2, 1);
          grads0(i, j, 3, 1) = grads1(i, j, 3, 1);
          grads0(i, j, 4, 1) = grads1(i, j, 4, 1);

          grads0(i, j, 0, 2) = -grads1(i, j, 0, 2);
          grads0(i, j, 1, 2) = grads1(i, j, 1, 2);
          grads0(i, j, 2, 2) = grads1(i, j, 2, 2);
          grads0(i, j, 3, 2) = grads1(i, j, 3, 2);
          grads0(i, j, 4, 2) = grads1(i, j, 4, 2);

          // negatespecies gradient (so gradient evaluates to zero
          // on wall)
          for (int n = 5; n < ne; n++) {
            grads0(i, j, n, 0) = -grads1(i, j, n, 0);
            grads0(i, j, n, 1) = -grads1(i, j, n, 1);
            grads0(i, j, n, 2) = -grads1(i, j, n, 2);
          }
        });
  }
}

PG_ABI void pgIsoTSlipWall(const pgView *q_, const pgView *Q_,
                           const pgView *qh_, const pgView *grads_,
                           const pgView *S_, const pgView *qBcVals_,
                           const pgView *QBcVals_, const pgView *rot_,
                           const pgDims *d, int nface, int terms, double tme) {
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

  if (terms == 0) {

    auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
    auto sVec = getFaceSlice(S, nface, blockFaceIdx);

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < ng; g++) {
      firstHaloIdx -= plus * g;

      auto q0 = getFaceSlice(q, nface, firstHaloIdx);

      Kokkos::parallel_for(
          "isoT slip wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            double S, nx, ny, nz;
            faceNormal(sVec(i, j, 0), sVec(i, j, 1), sVec(i, j, 2), S, nx, ny,
                       nz);

            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // flip velo on wall
            double uDotn =
                q1(i, j, 1) * nx + q1(i, j, 2) * ny + q1(i, j, 3) * nz;
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz;

            // set temperature
            q0(i, j, 4) = qBcVals(i, j, 4);
            // match species
            for (int n = 5; n < ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }

  } else if (terms == 2) {

    auto grads0 = getFaceSlice(grads, nface, firstHaloIdx);

    auto grads1 = getFaceSlice(grads, nface, firstInteriorCellIdx);

    MDRange2 range_face =
        MDRange2({0, 0}, {grads1.extent(0), grads1.extent(1)});
    Kokkos::parallel_for(
        "isoT slip visc terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          // negate velocity gradients
          grads0(i, j, 0, 0) = -grads1(i, j, 0, 0);
          grads0(i, j, 1, 0) = -grads1(i, j, 1, 0);
          grads0(i, j, 2, 0) = -grads1(i, j, 2, 0);
          grads0(i, j, 3, 0) = -grads1(i, j, 3, 0);

          grads0(i, j, 0, 1) = -grads1(i, j, 0, 1);
          grads0(i, j, 1, 1) = -grads1(i, j, 1, 1);
          grads0(i, j, 2, 1) = -grads1(i, j, 2, 1);
          grads0(i, j, 3, 1) = -grads1(i, j, 3, 1);

          grads0(i, j, 0, 2) = -grads1(i, j, 0, 2);
          grads0(i, j, 1, 2) = -grads1(i, j, 1, 2);
          grads0(i, j, 2, 2) = -grads1(i, j, 2, 2);
          grads0(i, j, 3, 2) = -grads1(i, j, 3, 2);

          // neumann temp gradients
          grads0(i, j, 4, 0) = grads1(i, j, 4, 0);
          grads0(i, j, 4, 1) = grads1(i, j, 4, 1);
          grads0(i, j, 4, 2) = grads1(i, j, 4, 2);

          // negate species gradient (so gradient evaluates to zero
          // on wall)
          for (int n = 5; n < ne; n++) {
            grads0(i, j, n, 0) = -grads1(i, j, n, 0);
            grads0(i, j, n, 1) = -grads1(i, j, n, 1);
            grads0(i, j, n, 2) = -grads1(i, j, n, 2);
          }
        });
  }
}

PG_ABI void pgIsoTMovingWall(const pgView *q_, const pgView *Q_,
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

  if (terms == 0) {

    auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
    auto sVec = getFaceSlice(S, nface, blockFaceIdx);

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < ng; g++) {
      firstHaloIdx -= plus * g;

      auto q0 = getFaceSlice(q, nface, firstHaloIdx);

      Kokkos::parallel_for(
          "Iso T moving wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            double S, nx, ny, nz;
            faceNormal(sVec(i, j, 0), sVec(i, j, 1), sVec(i, j, 2), S, nx, ny,
                       nz);

            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn =
                q1(i, j, 1) * nx + q1(i, j, 2) * ny + q1(i, j, 3) * nz;
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz;

            // set temperature
            q0(i, j, 4) = qBcVals(i, j, 4);
            // match species
            for (int n = 5; n < ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }

  } else if (terms == 1) {

    auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < ng; g++) {
      firstHaloIdx -= plus * g;

      auto q0 = getFaceSlice(q, nface, firstHaloIdx);

      Kokkos::parallel_for(
          "Iso T moving wall preDqDxyz terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // apply velo on wall
            q0(i, j, 1) = 2.0 * qBcVals(i, j, 1) - q1(i, j, 1);
            q0(i, j, 2) = 2.0 * qBcVals(i, j, 2) - q1(i, j, 2);
            q0(i, j, 3) = 2.0 * qBcVals(i, j, 3) - q1(i, j, 3);
          });
    }
  } else if (terms == 2) {

    auto grads0 = getFaceSlice(grads, nface, firstHaloIdx);

    auto grads1 = getFaceSlice(grads, nface, firstInteriorCellIdx);

    MDRange2 range_face =
        MDRange2({0, 0}, {grads1.extent(0), grads1.extent(1)});
    Kokkos::parallel_for(
        "Iso T moving wall postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          // negate pressure gradient, neumann velocity, temperature gradients
          grads0(i, j, 0, 0) = -grads1(i, j, 0, 0);
          grads0(i, j, 1, 0) = grads1(i, j, 1, 0);
          grads0(i, j, 2, 0) = grads1(i, j, 2, 0);
          grads0(i, j, 3, 0) = grads1(i, j, 3, 0);
          grads0(i, j, 4, 0) = grads1(i, j, 4, 0);

          grads0(i, j, 0, 1) = -grads1(i, j, 0, 1);
          grads0(i, j, 1, 1) = grads1(i, j, 1, 1);
          grads0(i, j, 2, 1) = grads1(i, j, 2, 1);
          grads0(i, j, 3, 1) = grads1(i, j, 3, 1);
          grads0(i, j, 4, 1) = grads1(i, j, 4, 1);

          grads0(i, j, 0, 2) = -grads1(i, j, 0, 2);
          grads0(i, j, 1, 2) = grads1(i, j, 1, 2);
          grads0(i, j, 2, 2) = grads1(i, j, 2, 2);
          grads0(i, j, 3, 2) = grads1(i, j, 3, 2);
          grads0(i, j, 4, 2) = grads1(i, j, 4, 2);

          // negate species gradient (so gradient evaluates to zero on wall)
          for (int n = 5; n < ne; n++) {
            grads0(i, j, n, 0) = -grads1(i, j, n, 0);
            grads0(i, j, n, 1) = -grads1(i, j, n, 1);
            grads0(i, j, n, 2) = -grads1(i, j, n, 2);
          }
        });
  }
}
