#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"

void centralDifference(block_ &b) {

  //-------------------------------------------------------------------------------------------|
  // i flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_i({b.ng, b.ng, b.ng},
                   {b.ni + b.ng, b.nj + b.ng - 1, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "2nd order central difference i face conv fluxes", range_i,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Compute face normal volume flux vector
        double uR = b.q(i, j, k, 1);
        double uL = b.q(i - 1, j, k, 1);

        double rhouR = b.Q(i, j, k, 1);
        double rhouL = b.Q(i - 1, j, k, 1);

        double a = b.isx(i, j, k);

        double pR = b.q(i, j, k, 0);
        double pL = b.q(i - 1, j, k, 0);

        // Compute fluxes

        // Continuity rho*Ui
        b.iF(i, j, k, 0) = 0.5 * (rhouR + rhouL) * a;

        // x momentum rho*u*Ui+ p*Ax
        b.iF(i, j, k, 1) = 0.5 * ((rhouR * uR + pR) + (rhouL * uL + pL)) * a;

        // y momentum rho*v*Ui+ p*Ay
        b.iF(i, j, k, 2) = 0.0;

        // w momentum rho*w*Ui+ p*Az
        b.iF(i, j, k, 3) = 0.0;

        // Total energy (rhoE+ p)*Ui)
        double rhoER = b.Q(i, j, k, 4);
        double rhoEL = b.Q(i - 1, j, k, 4);

        b.iF(i, j, k, 4) = 0.5 * ((rhoER + pR) * uR + (rhoEL + pL) * uL) * a;

        // Species
        for (int n = 0; n < b.ne - 5; n++) {
          b.iF(i, j, k, 5 + n) =
              0.5 * (b.Q(i, j, k, 5 + n) * uR + b.Q(i - 1, j, k, 5 + n) * uL) *
              a;
        }
      });

  //-------------------------------------------------------------------------------------------|
  // j flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_j({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "2nd order KEEP j face conv fluxes", range_j,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double V;
        double uf, rhouf;
        double vf, rhovf;
        double wf, rhowf;
        double pf;

        // Compute face normal volume flux vector
        uf = 0.5 * (b.q(i, j, k, 1) + b.q(i, j - 1, k, 1));
        vf = 0.5 * (b.q(i, j, k, 2) + b.q(i, j - 1, k, 2));
        wf = 0.5 * (b.q(i, j, k, 3) + b.q(i, j - 1, k, 3));

        V = b.jsx(i, j, k) * uf + b.jsy(i, j, k) * vf + b.jsz(i, j, k) * wf;

        pf = 0.5 * (b.q(i, j, k, 0) + b.q(i, j - 1, k, 0));

        rhouf = 0.5 * (b.Q(i, j, k, 1) + b.Q(i, j - 1, k, 1));
        rhovf = 0.5 * (b.Q(i, j, k, 2) + b.Q(i, j - 1, k, 2));
        rhowf = 0.5 * (b.Q(i, j, k, 3) + b.Q(i, j - 1, k, 3));

        // Compute fluxes
        double rho;
        rho = 0.5 * (b.Q(i, j, k, 0) + b.Q(i, j - 1, k, 0));

        // Continuity rho*Vj
        b.jF(i, j, k, 0) = rho * V;

        // x momentum rho*u*Vj+ pAx
        b.jF(i, j, k, 1) = rhouf * V + pf * b.jsx(i, j, k);

        // y momentum rho*v*Vj+ pAy
        b.jF(i, j, k, 2) = rhovf * V + pf * b.jsy(i, j, k);

        // w momentum rho*w*Vj+ pAz
        b.jF(i, j, k, 3) = rhowf * V + pf * b.jsz(i, j, k);

        // Total energy (rhoE+P)*Vj)
        double rhoE = 0.5 * (b.Q(i, j, k, 4) + b.Q(i, j - 1, k, 4));

        b.jF(i, j, k, 4) = (rhoE + pf) * V;

        // Species
        for (int n = 0; n < b.ne - 5; n++) {
          b.jF(i, j, k, 5 + n) =
              0.5 * (b.Q(i, j, k, 5 + n) + b.Q(i, j - 1, k, 5 + n)) * V;
        }
      });

  //-------------------------------------------------------------------------------------------|
  // k flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_k({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng});
  Kokkos::parallel_for(
      "2nd order KEEP k face conv fluxes", range_k,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double W;
        double uf, rhouf;
        double vf, rhovf;
        double wf, rhowf;
        double pf;

        // Compute face normal volume flux vector
        uf = 0.5 * (b.q(i, j, k, 1) + b.q(i, j, k - 1, 1));
        vf = 0.5 * (b.q(i, j, k, 2) + b.q(i, j, k - 1, 2));
        wf = 0.5 * (b.q(i, j, k, 3) + b.q(i, j, k - 1, 3));

        W = b.ksx(i, j, k) * uf + b.ksy(i, j, k) * vf + b.ksz(i, j, k) * wf;

        pf = 0.5 * (b.q(i, j, k, 0) + b.q(i, j, k - 1, 0));

        rhouf = 0.5 * (b.Q(i, j, k, 1) + b.Q(i, j, k - 1, 1));
        rhovf = 0.5 * (b.Q(i, j, k, 2) + b.Q(i, j, k - 1, 2));
        rhowf = 0.5 * (b.Q(i, j, k, 3) + b.Q(i, j, k - 1, 3));

        // Compute fluxes
        double rho;
        rho = 0.5 * (b.Q(i, j, k, 0) + b.Q(i, j, k - 1, 0));
        // Continuity rho*Wk
        b.kF(i, j, k, 0) = rho * W;

        // x momentum rho*u*Wk+ pAx
        b.kF(i, j, k, 1) = rhouf * W + pf * b.ksx(i, j, k);

        // y momentum rho*v*Wk+ pAy
        b.kF(i, j, k, 2) = rhovf * W + pf * b.ksy(i, j, k);

        // w momentum rho*w*Wk+ pAz
        b.kF(i, j, k, 3) = rhowf * W + pf * b.ksz(i, j, k);

        // Total energy (rhoE+P)*Wk)
        double rhoE = 0.5 * (b.Q(i, j, k, 4) + b.Q(i, j, k - 1, 4));

        b.kF(i, j, k, 4) = (rhoE + pf) * W;

        // Species
        for (int n = 0; n < b.ne - 5; n++) {
          b.kF(i, j, k, 5 + n) =
              0.5 * (b.Q(i, j, k, 5 + n) + b.Q(i, j, k - 1, 5 + n)) * W;
        }
      });
}
