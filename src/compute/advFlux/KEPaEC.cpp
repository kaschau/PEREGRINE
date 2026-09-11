#include "block_.hpp"
#include "kokkosTypes.hpp"

static void computeFlux(const block_ &b, fourDview &iF, const fourDview &iS,
                        const int iMod, const int jMod, const int kMod) {

  // face flux range
  MDRange3 range(
      {b.ng, b.ng, b.ng},
      {b.ni + b.ng - 1 + iMod, b.nj + b.ng - 1 + jMod, b.nk + b.ng - 1 + kMod});

  Kokkos::parallel_for(
      "2nd order KEPaEC conv fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Compute face normal volume flux vector
        double uf =
            0.5 * (b.q(i, j, k, 1) + b.q(i - iMod, j - jMod, k - kMod, 1));
        double vf =
            0.5 * (b.q(i, j, k, 2) + b.q(i - iMod, j - jMod, k - kMod, 2));
        double wf =
            0.5 * (b.q(i, j, k, 3) + b.q(i - iMod, j - jMod, k - kMod, 3));

        double U =
            iS(i, j, k, 0) * uf + iS(i, j, k, 1) * vf + iS(i, j, k, 2) * wf;

        double pf =
            0.5 * (b.q(i, j, k, 0) + b.q(i - iMod, j - jMod, k - kMod, 0));

        // Compute fluxes
        double rho =
            0.5 * (b.Q(i, j, k, 0) + b.Q(i - iMod, j - jMod, k - kMod, 0));

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
                    (b.q(i, j, k, 1) * b.q(i - iMod, j - jMod, k - kMod, 1) +
                     b.q(i, j, k, 2) * b.q(i - iMod, j - jMod, k - kMod, 2) +
                     b.q(i, j, k, 3) * b.q(i - iMod, j - jMod, k - kMod, 3));

        double Pj =
            0.5 * (b.q(i - iMod, j - jMod, k - kMod, 0) *
                       (b.q(i, j, k, 1) * iS(i, j, k, 0) +
                        b.q(i, j, k, 2) * iS(i, j, k, 1) +
                        b.q(i, j, k, 3) * iS(i, j, k, 2)) +
                   b.q(i, j, k, 0) *
                       (b.q(i - iMod, j - jMod, k - kMod, 1) * iS(i, j, k, 0) +
                        b.q(i - iMod, j - jMod, k - kMod, 2) * iS(i, j, k, 1) +
                        b.q(i - iMod, j - jMod, k - kMod, 3) * iS(i, j, k, 2)));

        // solve for internal energy flux
        double eR = b.qh(i, j, k, 4) / b.Q(i, j, k, 0);
        double eL = b.qh(i - iMod, j - jMod, k - kMod, 4) /
                    b.Q(i - iMod, j - jMod, k - kMod, 0);
        double Ij = 2.0 * (eL * eR) / (eL + eR) * C;

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

void KEPaEC(block_ &b) {
  computeFlux(b, b.iF, b.iS, 1, 0, 0);
  computeFlux(b, b.jF, b.jS, 0, 1, 0);
  computeFlux(b, b.kF, b.kS, 0, 0, 1);
};
