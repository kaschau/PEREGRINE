#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include <Kokkos_Core.hpp>

static void computeFlux(const unmanaged<double ****> &Q,
                        const unmanaged<double ****> &phi,
                        const unmanaged<double ****> &q,
                        const unmanaged<double ****> &qh, const int ne,
                        const int ng, const int ni, const int nj, const int nk,
                        unmanaged<double ****> &iF,
                        const unmanaged<double ****> &iS, const int iMod,
                        const int jMod, const int kMod) {

  const double kappa2 = 0.5;
  const double kappa4 = 0.005;

  // face flux range
  MDRange3 range({ng, ng, ng},
                 {ni + ng - 1 + iMod, nj + ng - 1 + jMod, nk + ng - 1 + kMod});

  Kokkos::parallel_for(
      "Scalar Dissipation face conv fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double S, nx, ny, nz;
        faceNormal(iS(i, j, k, 0), iS(i, j, k, 1), iS(i, j, k, 2), S, nx, ny,
                   nz);

        // The weird mod indexing math is so we grab the correct last
        // phi index for each dimension
        const int phiIndex =
            (iMod - 1) * iMod + (jMod)*jMod + (kMod + 1) * kMod;
        const double eps2 =
            kappa2 * fmax(phi(i, j, k, phiIndex),
                          phi(i - iMod, j - jMod, k - kMod, phiIndex));
        const double eps4 = fmax(0.0, kappa4 - eps2);

        // Compute face normal volume flux vector
        const double uf =
            0.5 * (q(i, j, k, 1) + q(i - iMod, j - jMod, k - kMod, 1));
        const double vf =
            0.5 * (q(i, j, k, 2) + q(i - iMod, j - jMod, k - kMod, 2));
        const double wf =
            0.5 * (q(i, j, k, 3) + q(i - iMod, j - jMod, k - kMod, 3));

        const double U = nx * uf + ny * vf + nz * wf;

        const double a =
            (abs(U) +
             0.5 * (qh(i, j, k, 3) + qh(i - iMod, j - jMod, k - kMod, 3))) *
            S;

        double rho2, rho4;
        rho2 = Q(i, j, k, 0) - Q(i - iMod, j - jMod, k - kMod, 0);
        rho4 = Q(i + iMod, j + jMod, k + kMod, 0) - 3.0 * Q(i, j, k, 0) +
               3.0 * Q(i - iMod, j - jMod, k - kMod, 0) -
               Q(i - iMod * 2, j - jMod * 2, k - kMod * 2, 0);

        // Continuity dissipation
        iF(i, j, k, 0) = a * (eps2 * rho2 - eps4 * rho4);

        // u momentum dissipation
        double u2, u4;
        u2 = Q(i, j, k, 1) - Q(i - iMod, j - jMod, k - kMod, 1);
        u4 = Q(i + iMod, j + jMod, k + kMod, 1) - 3.0 * Q(i, j, k, 1) +
             3.0 * Q(i - iMod, j - jMod, k - kMod, 1) -
             Q(i - iMod * 2, j - jMod * 2, k - kMod * 2, 1);

        iF(i, j, k, 1) = a * (eps2 * u2 - eps4 * u4);

        // v momentum dissipation
        double v2, v4;
        v2 = Q(i, j, k, 2) - Q(i - iMod, j - jMod, k - kMod, 2);
        v4 = Q(i + iMod, j + jMod, k + kMod, 2) - 3.0 * Q(i, j, k, 2) +
             3.0 * Q(i - iMod, j - jMod, k - kMod, 2) -
             Q(i - iMod * 2, j - jMod * 2, k - kMod * 2, 2);

        iF(i, j, k, 2) = a * (eps2 * v2 - eps4 * v4);

        // w momentum dissipation
        double w2, w4;
        w2 = Q(i, j, k, 3) - Q(i - iMod, j - jMod, k - kMod, 3);
        w4 = Q(i + iMod, j + jMod, k + kMod, 3) - 3.0 * Q(i, j, k, 3) +
             3.0 * Q(i - iMod, j - jMod, k - kMod, 3) -
             Q(i - iMod * 2, j - jMod * 2, k - kMod * 2, 3);

        iF(i, j, k, 3) = a * (eps2 * w2 - eps4 * w4);

        // total energy dissipation
        double e2, e4;
        e2 = Q(i, j, k, 4) - Q(i - iMod, j - jMod, k - kMod, 4);
        e4 = Q(i + iMod, j + jMod, k + kMod, 4) - 3.0 * Q(i, j, k, 4) +
             3.0 * Q(i - iMod, j - jMod, k - kMod, 4) -
             Q(i - iMod * 2, j - jMod * 2, k - kMod * 2, 4);

        iF(i, j, k, 4) = a * (eps2 * e2 - eps4 * e4);

        // Species
        for (int n = 0; n < ne - 5; n++) {
          double Y2, Y4;
          Y2 = Q(i, j, k, 5 + n) - Q(i - iMod, j - jMod, k - kMod, 5 + n);
          Y4 = Q(i + iMod, j + jMod, k + kMod, 5 + n) -
               3.0 * Q(i, j, k, 5 + n) +
               3.0 * Q(i - iMod, j - jMod, k - kMod, 5 + n) -
               Q(i - iMod * 2, j - jMod * 2, k - kMod * 2, 5 + n);
          iF(i, j, k, 5 + n) = a * (eps2 * Y2 - eps4 * Y4);
        }
      });
}

PG_ABI void pgScalarDissipation(const pgView *Q_, const pgView *iF_,
                                const pgView *iS_, const pgView *jF_,
                                const pgView *jS_, const pgView *kF_,
                                const pgView *kS_, const pgView *phi_,
                                const pgView *q_, const pgView *qh_,
                                const pgDims *d) {
  auto Q = as4(*Q_);
  auto iF = as4(*iF_);
  auto iS = as4(*iS_);
  auto jF = as4(*jF_);
  auto jS = as4(*jS_);
  auto kF = as4(*kF_);
  auto kS = as4(*kS_);
  auto phi = as4(*phi_);
  auto q = as4(*q_);
  auto qh = as4(*qh_);
  const int ng = d->ng, ni = d->ni, nj = d->nj, nk = d->nk;
  const int ne = Q.extent(3);
  computeFlux(Q, phi, q, qh, ne, ng, ni, nj, nk, iF, iS, 1, 0, 0);
  computeFlux(Q, phi, q, qh, ne, ng, ni, nj, nk, jF, jS, 0, 1, 0);
  computeFlux(Q, phi, q, qh, ne, ng, ni, nj, nk, kF, kS, 0, 0, 1);
}
