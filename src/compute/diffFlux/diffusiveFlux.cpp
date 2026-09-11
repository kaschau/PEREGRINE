#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include "thtrdat_.hpp"
#include <Kokkos_Core.hpp>

// This should basically never be used. It always worse than alpha damping.

static void computeFlux(const block_ &b, fourDview &iF, const fourDview &iS,
                        const int iMod, const int jMod, const int kMod) {

  // Stokes hypothesis
  double const bulkVisc = 0.0;

  // face flux range
  MDRange3 range(
      {b.ng, b.ng, b.ng},
      {b.ni + b.ng - 1 + iMod, b.nj + b.ng - 1 + jMod, b.nk + b.ng - 1 + kMod});

  Kokkos::parallel_for(
      "i face visc fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double mu =
            0.5 * (b.qt(i, j, k, 0) + b.qt(i - iMod, j - jMod, k - kMod, 0));
        double kappa =
            0.5 * (b.qt(i, j, k, 1) + b.qt(i - iMod, j - jMod, k - kMod, 1));
        double lambda = bulkVisc - 2.0 / 3.0 * mu;

        // continuity
        iF(i, j, k, 0) = 0.0;

        // Derivatives on face
        double dudx = 0.5 * (b.grads(i, j, k, 1, 0) +
                             b.grads(i - iMod, j - jMod, k - kMod, 1, 0));
        double dvdx = 0.5 * (b.grads(i, j, k, 2, 0) +
                             b.grads(i - iMod, j - jMod, k - kMod, 2, 0));
        double dwdx = 0.5 * (b.grads(i, j, k, 3, 0) +
                             b.grads(i - iMod, j - jMod, k - kMod, 3, 0));

        double dudy = 0.5 * (b.grads(i, j, k, 1, 1) +
                             b.grads(i - iMod, j - jMod, k - kMod, 1, 1));
        double dvdy = 0.5 * (b.grads(i, j, k, 2, 1) +
                             b.grads(i - iMod, j - jMod, k - kMod, 2, 1));
        double dwdy = 0.5 * (b.grads(i, j, k, 3, 1) +
                             b.grads(i - iMod, j - jMod, k - kMod, 3, 1));

        double dudz = 0.5 * (b.grads(i, j, k, 1, 2) +
                             b.grads(i - iMod, j - jMod, k - kMod, 1, 2));
        double dvdz = 0.5 * (b.grads(i, j, k, 2, 2) +
                             b.grads(i - iMod, j - jMod, k - kMod, 2, 2));
        double dwdz = 0.5 * (b.grads(i, j, k, 3, 2) +
                             b.grads(i - iMod, j - jMod, k - kMod, 3, 2));

        double div = dudx + dvdy + dwdz;

        // x momentum
        double txx = -2.0 * mu * dudx - lambda * div;
        double txy = -mu * (dvdx + dudy);
        double txz = -mu * (dwdx + dudz);

        iF(i, j, k, 1) =
            txx * iS(i, j, k, 0) + txy * iS(i, j, k, 1) + txz * iS(i, j, k, 2);

        // y momentum
        double &tyx = txy;
        double tyy = -2.0 * mu * dvdy - lambda * div;
        double tyz = -mu * (dwdy + dvdz);

        iF(i, j, k, 2) =
            tyx * iS(i, j, k, 0) + tyy * iS(i, j, k, 1) + tyz * iS(i, j, k, 2);

        // z momentum
        double &tzx = txz;
        double &tzy = tyz;
        double tzz = -2.0 * mu * dwdz - lambda * div;

        iF(i, j, k, 3) =
            tzx * iS(i, j, k, 0) + tzy * iS(i, j, k, 1) + tzz * iS(i, j, k, 2);

        // energy
        //   heat conduction
        double dTdx = 0.5 * (b.grads(i, j, k, 4, 0) +
                             b.grads(i - iMod, j - jMod, k - kMod, 4, 0));
        double dTdy = 0.5 * (b.grads(i, j, k, 4, 1) +
                             b.grads(i - iMod, j - jMod, k - kMod, 4, 1));
        double dTdz = 0.5 * (b.grads(i, j, k, 4, 2) +
                             b.grads(i - iMod, j - jMod, k - kMod, 4, 2));

        double q = -kappa * (dTdx * iS(i, j, k, 0) + dTdy * iS(i, j, k, 1) +
                             dTdz * iS(i, j, k, 2));

        // flow work
        // Compute face normal volume flux vector
        double uf =
            0.5 * (b.q(i, j, k, 1) + b.q(i - iMod, j - jMod, k - kMod, 1));
        double vf =
            0.5 * (b.q(i, j, k, 2) + b.q(i - iMod, j - jMod, k - kMod, 2));
        double wf =
            0.5 * (b.q(i, j, k, 3) + b.q(i - iMod, j - jMod, k - kMod, 3));

        iF(i, j, k, 4) = -(uf * txx + vf * txy + wf * txz) * iS(i, j, k, 0) -
                         (uf * tyx + vf * tyy + wf * tyz) * iS(i, j, k, 1) -
                         (uf * tzx + vf * tzy + wf * tzz) * iS(i, j, k, 2) + q;

        // Species
        double Dk, Vc = 0.0;
        double gradYns = 0.0;
        double rho =
            0.5 * (b.Q(i, j, k, 0) + b.Q(i - iMod, j - jMod, k - kMod, 0));
        // Compute the species flux and correction term \sum(k=1,ns) Dk*gradYk
        for (int n = 0; n < b.ne - 5; n++) {
          Dk = 0.5 * (b.qt(i, j, k, 2 + n) +
                      b.qt(i - iMod, j - jMod, k - kMod, 2 + n));
          double dYdx = 0.5 * (b.grads(i, j, k, 5 + n, 0) +
                               b.grads(i - iMod, j - jMod, k - kMod, 5 + n, 0));
          double dYdy = 0.5 * (b.grads(i, j, k, 5 + n, 1) +
                               b.grads(i - iMod, j - jMod, k - kMod, 5 + n, 1));
          double dYdz = 0.5 * (b.grads(i, j, k, 5 + n, 2) +
                               b.grads(i - iMod, j - jMod, k - kMod, 5 + n, 2));

          double gradYk = (dYdx * iS(i, j, k, 0) + dYdy * iS(i, j, k, 1) +
                           dYdz * iS(i, j, k, 2));
          gradYns -= gradYk;
          Vc += Dk * gradYk;
          iF(i, j, k, 5 + n) = -rho * Dk * gradYk;
        }
        // Apply n=ns species to correction
        Dk = 0.5 * (b.qt(i, j, k, 2 + b.ne - 5) +
                    b.qt(i - iMod, j - jMod, k - kMod, 2 + b.ne - 5));
        Vc += Dk * gradYns;

        // Apply correction and species thermal flux
        double Yk, hk;
        double Yns = 1.0;
        for (int n = 0; n < b.ne - 5; n++) {
          Yk = 0.5 *
               (b.q(i, j, k, 5 + n) + b.q(i - iMod, j - jMod, k - kMod, 5 + n));
          Yns -= Yk;
          iF(i, j, k, 5 + n) += Yk * rho * Vc;
          // Species thermal diffusion
          hk = 0.5 * (b.qh(i, j, k, 5 + n) +
                      b.qh(i - iMod, j - jMod, k - kMod, 5 + n));
          iF(i, j, k, 4) += iF(i, j, k, 5 + n) * hk;
        }
        // Apply the n=ns species to thermal diffusion
        Yns = fmax(Yns, 0.0);
        hk = 0.5 *
             (b.qh(i, j, k, b.ne) + b.qh(i - iMod, j - jMod, k - kMod, b.ne));
        iF(i, j, k, 4) += (-rho * Dk * gradYns + Yns * rho * Vc) * hk;
      });
}

void diffusiveFlux(block_ &b) {
  computeFlux(b, b.iF, b.iS, 1, 0, 0);
  computeFlux(b, b.jF, b.jS, 0, 1, 0);
  computeFlux(b, b.kF, b.kS, 0, 0, 1);
}
