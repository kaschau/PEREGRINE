#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include <Kokkos_Core.hpp>

// This should basically never be used. It always worse than alpha damping.

static void computeFlux(const in4 &Q, const in5 &grads, const in4 &q,
                        const in4 &qh, const in4 &qt, const pgDims &d,
                        const out4 &iF, const in4 &iS, const int iMod,
                        const int jMod, const int kMod) {

  // Stokes hypothesis
  double const bulkVisc = 0.0;

  const int ni = d.ni, nj = d.nj, nk = d.nk;
  // face flux range
  MDRange3 range({ng, ng, ng},
                 {ni + ng - 1 + iMod, nj + ng - 1 + jMod, nk + ng - 1 + kMod});

  Kokkos::parallel_for(
      "i face visc fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double mu =
            0.5 * (qt(i, j, k, 0) + qt(i - iMod, j - jMod, k - kMod, 0));
        double kappa =
            0.5 * (qt(i, j, k, 1) + qt(i - iMod, j - jMod, k - kMod, 1));
        double lambda = bulkVisc - 2.0 / 3.0 * mu;

        // no mass diffuses, so the continuity flux is left alone

        // Derivatives on face
        double dudx = 0.5 * (grads(i, j, k, 1, 0) +
                             grads(i - iMod, j - jMod, k - kMod, 1, 0));
        double dvdx = 0.5 * (grads(i, j, k, 2, 0) +
                             grads(i - iMod, j - jMod, k - kMod, 2, 0));
        double dwdx = 0.5 * (grads(i, j, k, 3, 0) +
                             grads(i - iMod, j - jMod, k - kMod, 3, 0));

        double dudy = 0.5 * (grads(i, j, k, 1, 1) +
                             grads(i - iMod, j - jMod, k - kMod, 1, 1));
        double dvdy = 0.5 * (grads(i, j, k, 2, 1) +
                             grads(i - iMod, j - jMod, k - kMod, 2, 1));
        double dwdy = 0.5 * (grads(i, j, k, 3, 1) +
                             grads(i - iMod, j - jMod, k - kMod, 3, 1));

        double dudz = 0.5 * (grads(i, j, k, 1, 2) +
                             grads(i - iMod, j - jMod, k - kMod, 1, 2));
        double dvdz = 0.5 * (grads(i, j, k, 2, 2) +
                             grads(i - iMod, j - jMod, k - kMod, 2, 2));
        double dwdz = 0.5 * (grads(i, j, k, 3, 2) +
                             grads(i - iMod, j - jMod, k - kMod, 3, 2));

        double div = dudx + dvdy + dwdz;

        // x momentum
        double txx = -2.0 * mu * dudx - lambda * div;
        double txy = -mu * (dvdx + dudy);
        double txz = -mu * (dwdx + dudz);

        iF(i, j, k, 1) +=
            txx * iS(i, j, k, 0) + txy * iS(i, j, k, 1) + txz * iS(i, j, k, 2);

        // y momentum
        double &tyx = txy;
        double tyy = -2.0 * mu * dvdy - lambda * div;
        double tyz = -mu * (dwdy + dvdz);

        iF(i, j, k, 2) +=
            tyx * iS(i, j, k, 0) + tyy * iS(i, j, k, 1) + tyz * iS(i, j, k, 2);

        // z momentum
        double &tzx = txz;
        double &tzy = tyz;
        double tzz = -2.0 * mu * dwdz - lambda * div;

        iF(i, j, k, 3) +=
            tzx * iS(i, j, k, 0) + tzy * iS(i, j, k, 1) + tzz * iS(i, j, k, 2);

        // energy
        //   heat conduction
        double dTdx = 0.5 * (grads(i, j, k, 4, 0) +
                             grads(i - iMod, j - jMod, k - kMod, 4, 0));
        double dTdy = 0.5 * (grads(i, j, k, 4, 1) +
                             grads(i - iMod, j - jMod, k - kMod, 4, 1));
        double dTdz = 0.5 * (grads(i, j, k, 4, 2) +
                             grads(i - iMod, j - jMod, k - kMod, 4, 2));

        double heatFlux =
            -kappa * (dTdx * iS(i, j, k, 0) + dTdy * iS(i, j, k, 1) +
                      dTdz * iS(i, j, k, 2));

        // flow work
        // Compute face normal volume flux vector
        double uf = 0.5 * (q(i, j, k, 1) + q(i - iMod, j - jMod, k - kMod, 1));
        double vf = 0.5 * (q(i, j, k, 2) + q(i - iMod, j - jMod, k - kMod, 2));
        double wf = 0.5 * (q(i, j, k, 3) + q(i - iMod, j - jMod, k - kMod, 3));

        iF(i, j, k, 4) += -(uf * txx + vf * txy + wf * txz) * iS(i, j, k, 0) -
                          (uf * tyx + vf * tyy + wf * tyz) * iS(i, j, k, 1) -
                          (uf * tzx + vf * tzy + wf * tzz) * iS(i, j, k, 2) +
                          heatFlux;

        // Species
        double Dk, Vc = 0.0;
        double gradYns = 0.0;
        double rho = 0.5 * (Q(i, j, k, 0) + Q(i - iMod, j - jMod, k - kMod, 0));
        // Compute the species flux and correction term \sum(k=1,ns) Dk*gradYk
        for (int n = 0; n < ne - 5; n++) {
          Dk = 0.5 *
               (qt(i, j, k, 2 + n) + qt(i - iMod, j - jMod, k - kMod, 2 + n));
          double dYdx = 0.5 * (grads(i, j, k, 5 + n, 0) +
                               grads(i - iMod, j - jMod, k - kMod, 5 + n, 0));
          double dYdy = 0.5 * (grads(i, j, k, 5 + n, 1) +
                               grads(i - iMod, j - jMod, k - kMod, 5 + n, 1));
          double dYdz = 0.5 * (grads(i, j, k, 5 + n, 2) +
                               grads(i - iMod, j - jMod, k - kMod, 5 + n, 2));

          double gradYk = (dYdx * iS(i, j, k, 0) + dYdy * iS(i, j, k, 1) +
                           dYdz * iS(i, j, k, 2));
          gradYns -= gradYk;
          Vc += Dk * gradYk;
          // the species flux before its correction, and its enthalpy with it
          double Jk = -rho * Dk * gradYk;
          double hk = 0.5 * (qh(i, j, k, 5 + n) +
                             qh(i - iMod, j - jMod, k - kMod, 5 + n));
          iF(i, j, k, 5 + n) += Jk;
          iF(i, j, k, 4) += Jk * hk;
        }
        // Apply n=ns species to correction
        Dk = 0.5 * (qt(i, j, k, 2 + ne - 5) +
                    qt(i - iMod, j - jMod, k - kMod, 2 + ne - 5));
        Vc += Dk * gradYns;

        // Apply correction and species thermal flux
        double Yk, hk;
        double Yns = 1.0;
        for (int n = 0; n < ne - 5; n++) {
          Yk = 0.5 *
               (q(i, j, k, 5 + n) + q(i - iMod, j - jMod, k - kMod, 5 + n));
          Yns -= Yk;
          // the correction, and its enthalpy with it
          double corr = Yk * rho * Vc;
          hk = 0.5 *
               (qh(i, j, k, 5 + n) + qh(i - iMod, j - jMod, k - kMod, 5 + n));
          iF(i, j, k, 5 + n) += corr;
          iF(i, j, k, 4) += corr * hk;
        }
        // Apply the n=ns species to thermal diffusion
        Yns = fmax(Yns, 0.0);
        hk = 0.5 * (qh(i, j, k, ne) + qh(i - iMod, j - jMod, k - kMod, ne));
        iF(i, j, k, 4) += (-rho * Dk * gradYns + Yns * rho * Vc) * hk;
      });
}

PG_ABI void pgDiffusiveFlux(int count, pgIn *Q_, pgIn *grads_, pgOut *iF_,
                            pgIn *iS_, pgOut *jF_, pgIn *jS_, pgOut *kF_,
                            pgIn *kS_, pgIn *q_, pgIn *qh_, pgIn *qt_,
                            const pgDims *d) {
  for (int e = 0; e < count; e++) {
    auto Q = as4(Q_[e]);
    auto grads = as5(grads_[e]);
    auto iF = as4(iF_[e]);
    auto iS = as4(iS_[e]);
    auto jF = as4(jF_[e]);
    auto jS = as4(jS_[e]);
    auto kF = as4(kF_[e]);
    auto kS = as4(kS_[e]);
    auto q = as4(q_[e]);
    auto qh = as4(qh_[e]);
    auto qt = as4(qt_[e]);
    computeFlux(Q, grads, q, qh, qt, d[e], iF, iS, 1, 0, 0);
    computeFlux(Q, grads, q, qh, qt, d[e], jF, jS, 0, 1, 0);
    computeFlux(Q, grads, q, qh, qt, d[e], kF, kS, 0, 0, 1);
  }
}
