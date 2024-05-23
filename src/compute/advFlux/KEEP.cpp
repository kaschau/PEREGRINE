#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <Kokkos_Core.hpp>

static void computeFlux(const block_ &b, fourDview &iF, const threeDview &isx,
                        const threeDview &isy, const threeDview &isz,
                        const int iMod, const int jMod, const int kMod) {

  // face flux range
  MDRange3 range(
      {b.ng, b.ng, b.ng},
      {b.ni + b.ng - 1 + iMod, b.nj + b.ng - 1 + jMod, b.nk + b.ng - 1 + kMod});
  Kokkos::parallel_for(
      "2nd order KEEP face conv fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Compute face normal volume flux vector
        double uf =
            0.5 * (b.q(i, j, k, 1) + b.q(i - iMod, j - jMod, k - kMod, 1));
        double vf =
            0.5 * (b.q(i, j, k, 2) + b.q(i - iMod, j - jMod, k - kMod, 2));
        double wf =
            0.5 * (b.q(i, j, k, 3) + b.q(i - iMod, j - jMod, k - kMod, 3));

        double U = isx(i, j, k) * uf + isy(i, j, k) * vf + isz(i, j, k) * wf;

        double pf =
            0.5 * (b.q(i, j, k, 0) + b.q(i - iMod, j - jMod, k - kMod, 0));

        double rho =
            0.5 * (b.Q(i, j, k, 0) + b.Q(i - iMod, j - jMod, k - kMod, 0));

        // Compute fluxes
        // Continuity rho*Ui
        double C = rho * U;
        iF(i, j, k, 0) = C;

        // x momentum rho*u*Ui+ p*Ax
        iF(i, j, k, 1) = C * uf + pf * isx(i, j, k);

        // y momentum rho*v*Ui+ p*Ay
        iF(i, j, k, 2) = C * vf + pf * isy(i, j, k);

        // w momentum rho*w*Ui+ p*Az
        iF(i, j, k, 3) = C * wf + pf * isz(i, j, k);

        // Total energy (rhoE+ p)*Ui)
        double Kj = C * 0.5 *
                    (b.q(i, j, k, 1) * b.q(i - iMod, j - jMod, k - kMod, 1) +
                     b.q(i, j, k, 2) * b.q(i - iMod, j - jMod, k - kMod, 2) +
                     b.q(i, j, k, 3) * b.q(i - iMod, j - jMod, k - kMod, 3));

        double Pj =
            0.5 * (b.q(i - iMod, j - jMod, k - kMod, 0) *
                       (b.q(i, j, k, 1) * isx(i, j, k) +
                        b.q(i, j, k, 2) * isy(i, j, k) +
                        b.q(i, j, k, 3) * isz(i, j, k)) +
                   b.q(i, j, k, 0) *
                       (b.q(i - iMod, j - jMod, k - kMod, 1) * isx(i, j, k) +
                        b.q(i - iMod, j - jMod, k - kMod, 2) * isy(i, j, k) +
                        b.q(i - iMod, j - jMod, k - kMod, 3) * isz(i, j, k)));

        double e = b.qh(i, j, k, 4) / b.Q(i, j, k, 0);
        double em = b.qh(i - iMod, j - jMod, k - kMod, 4) /
                    b.Q(i - iMod, j - jMod, k - kMod, 0);
        double Ij = C * 0.5 * (e + em);

        iF(i, j, k, 4) = Ij + Kj + Pj;

        // Species
        for (int n = 0; n < b.ne - 5; n++) {
          iF(i, j, k, 5 + n) =
              0.5 *
              (b.q(i, j, k, 5 + n) + b.q(i - iMod, j - jMod, k - kMod, 5 + n)) *
              C;
        }
      });
}

void KEEP(block_ &b) {
  computeFlux(b, b.iF, b.isx, b.isy, b.isz, 1, 0, 0);
  computeFlux(b, b.jF, b.jsx, b.jsy, b.jsz, 0, 1, 0);
  computeFlux(b, b.kF, b.ksx, b.ksy, b.ksz, 0, 0, 1);
}
