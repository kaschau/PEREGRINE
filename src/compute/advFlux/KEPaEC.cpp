#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"

static void computeFlux(const unmanaged<double ****> &Q,
                        const unmanaged<double ****> &q,
                        const unmanaged<double ****> &qh, const pgDims &d,
                        unmanaged<double ****> &iF,
                        const unmanaged<double ****> &iS, const int iMod,
                        const int jMod, const int kMod) {

  const int ni = d.ni, nj = d.nj, nk = d.nk;
  // face flux range
  MDRange3 range({ng, ng, ng},
                 {ni + ng - 1 + iMod, nj + ng - 1 + jMod, nk + ng - 1 + kMod});

  Kokkos::parallel_for(
      "2nd order KEPaEC conv fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Compute face normal volume flux vector
        double uf = 0.5 * (q(i, j, k, 1) + q(i - iMod, j - jMod, k - kMod, 1));
        double vf = 0.5 * (q(i, j, k, 2) + q(i - iMod, j - jMod, k - kMod, 2));
        double wf = 0.5 * (q(i, j, k, 3) + q(i - iMod, j - jMod, k - kMod, 3));

        double U =
            iS(i, j, k, 0) * uf + iS(i, j, k, 1) * vf + iS(i, j, k, 2) * wf;

        double pf = 0.5 * (q(i, j, k, 0) + q(i - iMod, j - jMod, k - kMod, 0));

        // Compute fluxes
        double rho = 0.5 * (Q(i, j, k, 0) + Q(i - iMod, j - jMod, k - kMod, 0));

        // Continuity rho*Ui
        double C = rho * U;
        iF(i, j, k, 0) = C;

        // x momentum rho*u*Ui+ p*Ax
        iF(i, j, k, 1) = C * uf + pf * iS(i, j, k, 0);

        // y momentum rho*v*Ui+ p*Ay
        iF(i, j, k, 2) = C * vf + pf * iS(i, j, k, 1);

        // w momentum rho*w*Ui+ p*Az
        iF(i, j, k, 3) = C * wf + pf * iS(i, j, k, 2);

        // Total energy (rhoE+ p)*Ui)
        double Kj = C * 0.5 *
                    (q(i, j, k, 1) * q(i - iMod, j - jMod, k - kMod, 1) +
                     q(i, j, k, 2) * q(i - iMod, j - jMod, k - kMod, 2) +
                     q(i, j, k, 3) * q(i - iMod, j - jMod, k - kMod, 3));

        double Pj =
            0.5 * (q(i - iMod, j - jMod, k - kMod, 0) *
                       (q(i, j, k, 1) * iS(i, j, k, 0) +
                        q(i, j, k, 2) * iS(i, j, k, 1) +
                        q(i, j, k, 3) * iS(i, j, k, 2)) +
                   q(i, j, k, 0) *
                       (q(i - iMod, j - jMod, k - kMod, 1) * iS(i, j, k, 0) +
                        q(i - iMod, j - jMod, k - kMod, 2) * iS(i, j, k, 1) +
                        q(i - iMod, j - jMod, k - kMod, 3) * iS(i, j, k, 2)));

        // solve for internal energy flux
        double eR = qh(i, j, k, 4) / Q(i, j, k, 0);
        double eL = qh(i - iMod, j - jMod, k - kMod, 4) /
                    Q(i - iMod, j - jMod, k - kMod, 0);
        double Ij = 2.0 * (eL * eR) / (eL + eR) * C;

        iF(i, j, k, 4) = Ij + Kj + Pj;

        // Species
        for (int n = 0; n < ne - 5; n++) {
          iF(i, j, k, 5 + n) =
              0.5 *
              (q(i, j, k, 5 + n) + q(i - iMod, j - jMod, k - kMod, 5 + n)) * C;
        }
      });
}

PG_ABI void pgKEPaEC(int count, const pgView *Q_, const pgView *iF_,
                     const pgView *iS_, const pgView *jF_, const pgView *jS_,
                     const pgView *kF_, const pgView *kS_, const pgView *q_,
                     const pgView *qh_, const pgDims *d) {
  for (int e = 0; e < count; e++) {
    auto Q = as4(Q_[e]);
    auto iF = as4(iF_[e]);
    auto iS = as4(iS_[e]);
    auto jF = as4(jF_[e]);
    auto jS = as4(jS_[e]);
    auto kF = as4(kF_[e]);
    auto kS = as4(kS_[e]);
    auto q = as4(q_[e]);
    auto qh = as4(qh_[e]);
    computeFlux(Q, q, qh, d[e], iF, iS, 1, 0, 0);
    computeFlux(Q, q, qh, d[e], jF, jS, 0, 1, 0);
    computeFlux(Q, q, qh, d[e], kF, kS, 0, 0, 1);
  }
}
