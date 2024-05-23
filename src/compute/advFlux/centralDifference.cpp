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
      "2nd order central difference face conv fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Compute face normal volume flux vector
        double uR = b.q(i, j, k, 1);
        double uL = b.q(i - iMod, j - jMod, k - kMod, 1);
        double vR = b.q(i, j, k, 2);
        double vL = b.q(i - iMod, j - jMod, k - kMod, 2);
        double wR = b.q(i, j, k, 3);
        double wL = b.q(i - iMod, j - jMod, k - kMod, 3);

        double UfR = uR * isx(i, j, k) + vR * isy(i, j, k) + wR * isz(i, j, k);
        double UfL = uL * isx(i, j, k) + vL * isy(i, j, k) + wL * isz(i, j, k);

        double pR = b.q(i, j, k, 0);
        double pL = b.q(i - iMod, j - jMod, k - kMod, 0);
        // Compute fluxes

        // Continuity rho*Ui
        double rhouR = b.Q(i, j, k, 1);
        double rhovR = b.Q(i, j, k, 2);
        double rhowR = b.Q(i, j, k, 3);
        double rhouL = b.Q(i - iMod, j - jMod, k - kMod, 1);
        double rhovL = b.Q(i - iMod, j - jMod, k - kMod, 2);
        double rhowL = b.Q(i - iMod, j - jMod, k - kMod, 3);
        double CjR =
            isx(i, j, k) * rhouR + isy(i, j, k) * rhovR + isz(i, j, k) * rhowR;
        double CjL =
            isx(i, j, k) * rhouL + isy(i, j, k) * rhovL + isz(i, j, k) * rhowL;
        iF(i, j, k, 0) = 0.5 * (CjR + CjL);

        // x momentum rho*u*Ui+ p*Ax
        iF(i, j, k, 1) = 0.5 * (rhouR * UfR + pR * isx(i, j, k) + rhouL * UfL +
                                pL * isx(i, j, k));

        // y momentum rho*v*Ui+ p*Ay
        iF(i, j, k, 2) = 0.5 * (rhovR * UfR + pR * isy(i, j, k) + rhovL * UfL +
                                pL * isy(i, j, k));

        // w momentum rho*w*Ui+ p*Az
        iF(i, j, k, 3) = 0.5 * (rhowR * UfR + pR * isz(i, j, k) + rhowL * UfL +
                                pL * isz(i, j, k));

        // Total energy (rhoE+ p)*Ui)
        double rhoER = b.Q(i, j, k, 4);
        double rhoEL = b.Q(i - iMod, j - jMod, k - kMod, 4);

        iF(i, j, k, 4) = 0.5 * ((rhoER + pR) * UfR + (rhoEL + pL) * UfL);

        // Species
        for (int n = 0; n < b.ne - 5; n++) {
          iF(i, j, k, 5 + n) =
              0.5 * (b.Q(i, j, k, 5 + n) * UfR +
                     b.Q(i - iMod, j - jMod, k - kMod, 5 + n) * UfL);
        }
      });
}

void centralDifference(block_ &b) {
  computeFlux(b, b.iF, b.isx, b.isy, b.isz, 1, 0, 0);
  computeFlux(b, b.jF, b.jsx, b.jsy, b.jsz, 0, 1, 0);
  computeFlux(b, b.kF, b.ksx, b.ksy, b.ksz, 0, 0, 1);
};
