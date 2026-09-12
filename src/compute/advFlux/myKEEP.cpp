#include "abi.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

static void computeFlux(const unmanaged<double ****> &Q,
                        const unmanaged<double ****> &q,
                        const unmanaged<double ****> &qh, const int ne,
                        const int ng, const int ni, const int nj, const int nk,
                        unmanaged<double ****> &iF,
                        const unmanaged<double ****> &iS, const int iMod,
                        const int jMod, const int kMod) {

  // face flux range
  MDRange3 range({ng, ng, ng},
                 {ni + ng - 1 + iMod, nj + ng - 1 + jMod, nk + ng - 1 + kMod});

  Kokkos::parallel_for(
      "2nd order myKEEP face conv fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Compute face normal volume flux vector
        double uf = 0.5 * (q(i, j, k, 1) + q(i - iMod, j - jMod, k - kMod, 1));
        double vf = 0.5 * (q(i, j, k, 2) + q(i - iMod, j - jMod, k - kMod, 2));
        double wf = 0.5 * (q(i, j, k, 3) + q(i - iMod, j - jMod, k - kMod, 3));

        double U =
            iS(i, j, k, 0) * uf + iS(i, j, k, 1) * vf + iS(i, j, k, 2) * wf;

        double pf = 0.5 * (q(i, j, k, 0) + q(i - iMod, j - jMod, k - kMod, 0));

        // Compute fluxes
        double rho;
        rho = 0.5 * (Q(i, j, k, 0) + Q(i - iMod, j - jMod, k - kMod, 0));

        // Continuity rho*Ui
        double Cj = rho * U;
        iF(i, j, k, 0) = Cj;

        // x momentum rho*u*Ui+ p*Ax
        iF(i, j, k, 1) = Cj * uf + pf * iS(i, j, k, 0);

        // y momentum rho*v*Ui+ p*Ay
        iF(i, j, k, 2) = Cj * vf + pf * iS(i, j, k, 1);

        // w momentum rho*w*Ui+ p*Az
        iF(i, j, k, 3) = Cj * wf + pf * iS(i, j, k, 2);

        // Total energy (rhoE+ p)*Ui)
        double Kj = rho * 0.5 *
                    (q(i, j, k, 1) * q(i - iMod, j - jMod, k - kMod, 1) +
                     q(i, j, k, 2) * q(i - iMod, j - jMod, k - kMod, 2) +
                     q(i, j, k, 3) * q(i - iMod, j - jMod, k - kMod, 3)) *
                    U;

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
        double cvR = qh(i, j, k, 1) / qh(i, j, k, 0);
        double &TR = q(i, j, k, 4);
        double &rhoR = Q(i, j, k, 0);
        double RR = qh(i, j, k, 1) - cvR;
        double hR = qh(i, j, k, 2) / rhoR;
        double eR = qh(i, j, k, 4);
        double &uR = q(i, j, k, 1);
        double &vR = q(i, j, k, 2);
        double &wR = q(i, j, k, 3);
        double sR = cvR * log(TR) - RR * log(rhoR);
        double phiR =
            -RR * rhoR *
            (uR * iS(i, j, k, 0) + vR * iS(i, j, k, 1) + wR * iS(i, j, k, 2));

        double v0R =
            sR + (-hR + 0.5 * (pow(uR, 2) + pow(vR, 2) + pow(wR, 2))) / TR;
        double v1R = -uR / TR;
        double v2R = -vR / TR;
        double v3R = -wR / TR;
        double v4R = 1.0 / TR;

        // left
        double cvL = qh(i - iMod, j - jMod, k - kMod, 1) /
                     qh(i - iMod, j - jMod, k - kMod, 0);
        double &TL = q(i - iMod, j - jMod, k - kMod, 4);
        double &rhoL = Q(i - iMod, j - jMod, k - kMod, 0);
        double RL = qh(i - iMod, j - jMod, k - kMod, 1) - cvL;
        double hL = qh(i - iMod, j - jMod, k - kMod, 2) / rhoL;
        double eL = qh(i - iMod, j - jMod, k - kMod, 4);
        double &uL = q(i - iMod, j - jMod, k - kMod, 1);
        double &vL = q(i - iMod, j - jMod, k - kMod, 2);
        double &wL = q(i - iMod, j - jMod, k - kMod, 3);
        double sL = cvL * log(TL) - RL * log(rhoL);
        double phiL =
            -RL * rhoL *
            (uL * iS(i, j, k, 0) + vL * iS(i, j, k, 1) + wL * iS(i, j, k, 2));

        double v0L =
            sL + (-hL + 0.5 * (pow(uL, 2) + pow(vL, 2) + pow(wL, 2))) / TL;
        double v1L = -uL / TL;
        double v2L = -vL / TL;
        double v3L = -wL / TL;
        double v4L = 1.0 / TL;

        double V0 = 0.5 * (v0R + v0L);
        double V1 = 0.5 * (v1R + v1L);
        double V2 = 0.5 * (v2R + v2L);
        double V3 = 0.5 * (v3R + v3L);
        double V4 = 0.5 * (v4R + v4L);
        double PHI = 0.5 * (phiR + phiL);

        double Fs = rho * U * 0.5 * (sR + sL);

        double Ij = (Fs + PHI - V0 * iF(i, j, k, 0) - V1 * iF(i, j, k, 1) -
                     V2 * iF(i, j, k, 2) - V3 * iF(i, j, k, 3)) /
                        V4 -
                    Pj - Kj;

        iF(i, j, k, 4) = Ij + Kj + Pj;

        // Species
        for (int n = 0; n < ne - 5; n++) {
          iF(i, j, k, 5 + n) =
              0.5 *
              (Q(i, j, k, 5 + n) + Q(i - iMod, j - jMod, k - kMod, 5 + n)) * U;
        }
      });
}

PG_ABI void pgMyKEEP(const pgView *Q_, const pgView *iF_, const pgView *iS_,
                     const pgView *jF_, const pgView *jS_, const pgView *kF_,
                     const pgView *kS_, const pgView *q_, const pgView *qh_,
                     const pgDims *d) {
  auto Q = as4(*Q_);
  auto iF = as4(*iF_);
  auto iS = as4(*iS_);
  auto jF = as4(*jF_);
  auto jS = as4(*jS_);
  auto kF = as4(*kF_);
  auto kS = as4(*kS_);
  auto q = as4(*q_);
  auto qh = as4(*qh_);
  const int ng = d->ng, ni = d->ni, nj = d->nj, nk = d->nk;
  const int ne = Q.extent(3);
  computeFlux(Q, q, qh, ne, ng, ni, nj, nk, iF, iS, 1, 0, 0);
  computeFlux(Q, q, qh, ne, ng, ni, nj, nk, jF, jS, 0, 1, 0);
  computeFlux(Q, q, qh, ne, ng, ni, nj, nk, kF, kS, 0, 0, 1);
}
