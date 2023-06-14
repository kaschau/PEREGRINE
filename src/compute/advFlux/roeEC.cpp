#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"

void roeEC(block_ &b, const thtrdat_ &th) {

  //-------------------------------------------------------------------------------------------|
  // i flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_i({b.ng, b.ng, b.ng},
                   {b.ni + b.ng, b.nj + b.ng - 1, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "2nd order KEEP i face conv fluxes", range_i,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double &pL = b.q(i - 1, j, k, 0);
        double &uL = b.q(i - 1, j, k, 1);
        double &vL = b.q(i - 1, j, k, 2);
        double &wL = b.q(i - 1, j, k, 3);
        double &TL = b.q(i - 1, j, k, 4);
        double &rhoL = b.Q(i - 1, j, k, 0);

        double &pR = b.q(i, j, k, 0);
        double &uR = b.q(i, j, k, 1);
        double &vR = b.q(i, j, k, 2);
        double &wR = b.q(i, j, k, 3);
        double &TR = b.q(i, j, k, 4);
        double &rhoR = b.Q(i, j, k, 0);

        double rhoBar = 0.5 * (rhoR + rhoL);
        double uBar = 0.5 * (uR + uL);

        double &gamma = b.qh(i, j, k, 0);

        double z1R = sqrt(rhoR / pR);
        double z2R = sqrt(rhoR / pR) * uR;
        double z3R = sqrt(rhoR / pR) * pR;

        double z1L = sqrt(rhoL / pL);
        double z2L = sqrt(rhoL / pL) * uL;
        double z3L = sqrt(rhoL / pL) * pL;

        // Compute fluxes
        double eps = 0.01;

        double z1Bar = 0.5 * (z1L + z1R);
        double z2Bar = 0.5 * (z2L + z2R);
        double z3Bar = 0.5 * (z3L + z3R);

        // z1Hat
        double ksi = z1L / z1R;
        double f = (ksi - 1.0) / (ksi + 1.0);
        double fu = pow(f, 2.0);
        double F;
        if (fu < eps) {
          F = 1.0 + fu / 3.0 + fu * fu / 5.0 + fu * fu * fu / 7.0 +
              fu * fu * fu * fu / 9.0;
        } else {
          F = log(ksi) / 2.0 / fu;
        }
        double z1Hat = (z1R + z1L) / (2 * F);

        // z3Hat
        ksi = z3L / z3R;
        f = (ksi - 1.0) / (ksi + 1.0);
        fu = pow(fu, 2.0);
        if (fu < eps) {
          F = 1.0 + fu / 3.0 + fu * fu / 5.0 + fu * fu * fu / 7.0 +
              fu * fu * fu * fu / 9.0;
        } else {
          F = log(ksi) / 2.0 / fu;
        }
        double z3Hat = (z3R + z3L) / (2 * F);

        double rhoTilde = z1Bar * z3Hat;
        double uTilde = z2Bar / z1Bar;
        double p1Tilde = z3Bar / z1Bar;
        double p2Tilde = (gamma + 1.0) / (2.0 * gamma) * z3Hat / z1Hat +
                         (gamma - 1.0) / (2.0 * gamma) * z3Bar / z1Bar;
        double aTilde = sqrt(gamma * p2Tilde / rhoTilde);
        double HTilde =
            pow(aTilde, 2.0) / (gamma - 1.0) + 0.5 * uTilde * uTilde;

        // Continuity rho*Ui
        b.iF(i, j, k, 0) = rhoTilde * uTilde * b.iS(i, j, k);

        // x momentum rho*u*Ui+ p*Ax
        b.iF(i, j, k, 1) =
            (p1Tilde + uTilde * uTilde * rhoTilde) * b.iS(i, j, k);

        // y momentum rho*v*Ui+ p*Ay
        b.iF(i, j, k, 2) = 0.0;

        // w momentum rho*w*Ui+ p*Az
        b.iF(i, j, k, 3) = 0.0;

        // Total energy (rhoE+ p)*Ui)
        b.iF(i, j, k, 4) = HTilde * rhoTilde * uTilde * b.iS(i, j, k);
      });

  //-------------------------------------------------------------------------------------------|
  // j flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_j({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng, b.nk + b.ng - 1});
  Kokkos::parallel_for("2nd order KEEP j face conv fluxes", range_j,
                       KOKKOS_LAMBDA(const int i, const int j, const int k){});

  //-------------------------------------------------------------------------------------------|
  // k flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_k({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng});
  Kokkos::parallel_for("2nd order KEEP k face conv fluxes", range_k,
                       KOKKOS_LAMBDA(const int i, const int j, const int k){});
}
