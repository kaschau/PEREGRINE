#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <string.h>

PG_ABI void pgConstantVelocitySubsonicInlet(
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
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});

    for (int g = 0; g < ng; g++) {
      firstHaloIdx -= plus * g;
      secondInteriorCellIdx += plus * g;

      auto q0 = getFaceSlice(q, nface, firstHaloIdx);
      auto q2 = getFaceSlice(q, nface, secondInteriorCellIdx);

      Kokkos::parallel_for(
          "Constant velocity subsonic inlet euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // extrapolate pressure
            q0(i, j, 0) = 2.0 * q1(i, j, 0) - q2(i, j, 0);

            // apply velo in halo
            q0(i, j, 1) = qBcVals(i, j, 1);
            q0(i, j, 2) = qBcVals(i, j, 2);
            q0(i, j, 3) = qBcVals(i, j, 3);

            // apply temperature in halo
            q0(i, j, 4) = qBcVals(i, j, 4);

            // apply species in halo
            for (int n = 5; n < ne; n++) {
              q0(i, j, n) = qBcVals(i, j, n);
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
        "Constant velocity subsonic inlet postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j, const int l) {
          // neumann all gradients
          for (int d = 0; d < 3; d++) {
            grads0(i, j, l, d) = grads1(i, j, l, d);
          }
        });
  }
}
PG_ABI void pgSupersonicInlet(const pgView *q_, const pgView *Q_,
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
    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<int>(q1.extent(0)),
                             static_cast<int>(q1.extent(1)), ne});

    for (int g = 0; g < ng; g++) {
      firstHaloIdx -= plus * g;
      secondInteriorCellIdx += plus * g;

      auto q0 = getFaceSlice(q, nface, firstHaloIdx);
      auto q2 = getFaceSlice(q, nface, secondInteriorCellIdx);

      Kokkos::parallel_for(
          "Supersonic inlet euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j, const int l) {
            // apply all variables on face
            q0(i, j, l) = qBcVals(i, j, l);
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
        "Supersonic inlet postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j, const int l) {
          // neumann all gradients
          for (int d = 0; d < 3; d++) {
            grads0(i, j, l, d) = grads1(i, j, l, d);
          }
        });
  }
}

PG_ABI void pgConstantMassFluxSubsonicInlet(
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
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});

    for (int g = 0; g < ng; g++) {
      firstHaloIdx -= plus * g;
      secondInteriorCellIdx += plus * g;

      auto q0 = getFaceSlice(q, nface, firstHaloIdx);
      auto q2 = getFaceSlice(q, nface, secondInteriorCellIdx);

      Kokkos::parallel_for(
          "Constant mass flux subsonic inlet euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // extrapolate pressure
            q0(i, j, 0) = 2.0 * q1(i, j, 0) - q2(i, j, 0);

            // apply zero velo to halo to make subsequent updates easier
            q0(i, j, 1) = 0.0;
            q0(i, j, 2) = 0.0;
            q0(i, j, 3) = 0.0;

            // apply temperature to halo
            q0(i, j, 4) = qBcVals(i, j, 4);

            // apply species to halo
            for (int n = 5; n < ne; n++) {
              q0(i, j, n) = qBcVals(i, j, n);
            }
          });
    }
  } else if (terms == 3) {
    // the eos has run on the halo from python: density is valid
    auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    // set momentums, and velocities to match the desired mass flux
    // NOTE: We have to be careful with the indexing to accomodate fourth
    // order. In particular, we cannot just use firstInteriorCellIdx for all the
    // extrapolations so we have to make blockFaceIdx start with
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
            q0(i, j, 1) =
                4.0 * rhou / (Q0(i, j, 0) + Q2(i, j, 0)) - q2(i, j, 1);
            q0(i, j, 2) =
                4.0 * rhov / (Q0(i, j, 0) + Q2(i, j, 0)) - q2(i, j, 2);
            q0(i, j, 3) =
                4.0 * rhow / (Q0(i, j, 0) + Q2(i, j, 0)) - q2(i, j, 3);

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
  } else if (terms == 2) {

    // Only applied to first halo slice
    auto grads1 = getFaceSlice(grads, nface, firstInteriorCellIdx);

    auto grads0 = getFaceSlice(grads, nface, firstHaloIdx);

    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(grads1.extent(0)),
                             static_cast<long>(grads1.extent(1)), ne});
    Kokkos::parallel_for(
        "Supersonic inlet postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j, const int l) {
          // neumann all gradients
          for (int d = 0; d < 3; d++) {
            grads0(i, j, l, d) = grads1(i, j, l, d);
          }
        });
  }
}

PG_ABI void pgStagnationSubsonicInlet(const pgView *q_, const pgView *Q_,
                                      const pgView *qh_, const pgView *grads_,
                                      const pgView *S_, const pgView *qBcVals_,
                                      const pgView *QBcVals_,
                                      const pgView *rot_, const pgDims *d,
                                      int nface, int terms, double tme) {
  auto q = as4(*q_), Q = as4(*Q_), qh = as4(*qh_);
  auto grads = as5(*grads_);
  auto S = as4(*S_);
  auto qBcVals = as3(*qBcVals_), QBcVals = as3(*QBcVals_);
  auto rot = as2(*rot_);
  const int ni = d->ni, nj = d->nj, nk = d->nk, ng = d->ng;
  const int ne = q.extent(3);

  // Stagnation boundary condition from
  // https://ntrs.nasa.gov/api/citations/20180001221/downloads/20180001221.pdf

  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const faceCells f = faceCellsOf(*d, nface);
  int firstHaloIdx = f.halo, firstInteriorCellIdx = f.interior,
      blockFaceIdx = f.face, plus = f.plus;

  if (terms == 0) {

    auto q1 = getFaceSlice(q, nface, firstInteriorCellIdx);
    auto Q1 = getFaceSlice(Q, nface, firstInteriorCellIdx);
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

  } else if (terms == 2) {

    // Only applied to first halo slice
    auto grads1 = getFaceSlice(grads, nface, firstInteriorCellIdx);

    auto grads0 = getFaceSlice(grads, nface, firstHaloIdx);

    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(grads1.extent(0)),
                             static_cast<long>(grads1.extent(1)), ne});
    Kokkos::parallel_for(
        "Supersonic inlet postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j, const int l) {
          // neumann all gradients
          for (int d = 0; d < 3; d++) {
            grads0(i, j, l, d) = grads1(i, j, l, d);
          }
        });
  }
}
