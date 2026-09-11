#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <Kokkos_Core.hpp>

static void computeFlux(const block_ &b, fourDview &iF, const fourDview &iS,
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

        double UfR =
            uR * iS(i, j, k, 0) + vR * iS(i, j, k, 1) + wR * iS(i, j, k, 2);
        double UfL =
            uL * iS(i, j, k, 0) + vL * iS(i, j, k, 1) + wL * iS(i, j, k, 2);

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
        double CjR = iS(i, j, k, 0) * rhouR + iS(i, j, k, 1) * rhovR +
                     iS(i, j, k, 2) * rhowR;
        double CjL = iS(i, j, k, 0) * rhouL + iS(i, j, k, 1) * rhovL +
                     iS(i, j, k, 2) * rhowL;
        iF(i, j, k, 0) = 0.5 * (CjR + CjL);

        // x momentum rho*u*Ui+ p*Ax
        iF(i, j, k, 1) = 0.5 * (rhouR * UfR + pR * iS(i, j, k, 0) +
                                rhouL * UfL + pL * iS(i, j, k, 0));

        // y momentum rho*v*Ui+ p*Ay
        iF(i, j, k, 2) = 0.5 * (rhovR * UfR + pR * iS(i, j, k, 1) +
                                rhovL * UfL + pL * iS(i, j, k, 1));

        // w momentum rho*w*Ui+ p*Az
        iF(i, j, k, 3) = 0.5 * (rhowR * UfR + pR * iS(i, j, k, 2) +
                                rhowL * UfL + pL * iS(i, j, k, 2));

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
  computeFlux(b, b.iF, b.iS, 1, 0, 0);
  computeFlux(b, b.jF, b.jS, 0, 1, 0);
  computeFlux(b, b.kF, b.kS, 0, 0, 1);
};
