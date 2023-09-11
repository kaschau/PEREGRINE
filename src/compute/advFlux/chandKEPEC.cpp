#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"

void chandKEPEC(block_ &b) {

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
        double &rhoL = b.Q(i - 1, j, k, 0);
        double betaL = rhoL / (2.0 * pL);
        double kL = uL * uL + vL * vL + wL * wL;

        double &pR = b.q(i, j, k, 0);
        double &uR = b.q(i, j, k, 1);
        double &vR = b.q(i, j, k, 2);
        double &wR = b.q(i, j, k, 3);
        double &rhoR = b.Q(i, j, k, 0);
        double betaR = rhoR / (2.0 * pR);
        double kR = uR * uR + vR * vR + wR * wR;

        double rhoBar = (rhoR + rhoL) / 2.0;
        double betaBar = (betaR + betaL) / 2.0;
        double uBar = (uR + uL) / 2.0;
        double vBar = (vR + vL) / 2.0;
        double wBar = (wR + wL) / 2.0;
        double kBar = (kR + kL) / 2.0;

        double &gamma = b.qh(i, j, k, 0);
        double U = uBar * b.isx(i, j, k) + vBar * b.isy(i, j, k) +
                   wBar * b.isz(i, j, k);

        double eps = 0.01;
        // rhoHat
        double rhoHat;
        if (rhoL == rhoR) {
          double ksi = rhoL / rhoR;
          double f = (ksi - 1.0) / (ksi + 1.0);
          double fu = pow(f, 2.0);
          double F;
          if (fu < eps) {
            F = 1.0 + fu / 3.0 + fu * fu / 5.0 + fu * fu * fu / 7.0 +
                fu * fu * fu * fu / 9.0;
          } else {
            F = log(ksi) / 2.0 / fu;
          }
          rhoHat = (rhoR + rhoL) / (2 * F);
        } else {
          rhoHat = (rhoR - rhoL) / (log(rhoR) - log(rhoL));
        }
        // betaHat
        double betaHat;
        if (betaL == betaR) {
          double ksi = betaL / betaR;
          double f = (ksi - 1.0) / (ksi + 1.0);
          double fu = pow(f, 2.0);
          double F;
          if (fu < eps) {
            F = 1.0 + fu / 3.0 + fu * fu / 5.0 + fu * fu * fu / 7.0 +
                fu * fu * fu * fu / 9.0;
          } else {
            F = log(ksi) / 2.0 / fu;
          }
          betaHat = (betaR + betaL) / (2 * F);
        } else {
          betaHat = (betaR - betaL) / (log(betaR) - log(betaL));
        }

        double pTilde = rhoBar / (2.0 * betaBar);

        // Continuity rho*Ui
        b.iF(i, j, k, 0) = rhoHat * U;

        // x momentum rho*u*Ui+ p*Ax
        b.iF(i, j, k, 1) = b.iF(i, j, k, 0) * uBar + pTilde * b.isx(i, j, k);

        // y momentum rho*v*Ui+ p*Ay
        b.iF(i, j, k, 2) = b.iF(i, j, k, 0) * vBar + pTilde * b.isy(i, j, k);

        // w momentum rho*w*Ui+ p*Az
        b.iF(i, j, k, 3) = b.iF(i, j, k, 0) * wBar + pTilde * b.isz(i, j, k);

        // Total energy (rhoE+ p)*Ui)

        b.iF(i, j, k, 4) =
            (1.0 / (2.0 * (gamma - 1.0) * betaHat) - 0.5 * kBar) *
                b.iF(i, j, k, 0) +
            uBar * b.iF(i, j, k, 1) + vBar * b.iF(i, j, k, 2) +
            wBar * b.iF(i, j, k, 3);
      });

  //-------------------------------------------------------------------------------------------|
  // j flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_j({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "2nd order KEEP j face conv fluxes", range_j,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double &pL = b.q(i, j - 1, k, 0);
        double &uL = b.q(i, j - 1, k, 1);
        double &vL = b.q(i, j - 1, k, 2);
        double &wL = b.q(i, j - 1, k, 3);
        double &rhoL = b.Q(i, j - 1, k, 0);
        double betaL = rhoL / (2.0 * pL);
        double kL = uL * uL + vL * vL + wL * wL;

        double &pR = b.q(i, j, k, 0);
        double &uR = b.q(i, j, k, 1);
        double &vR = b.q(i, j, k, 2);
        double &wR = b.q(i, j, k, 3);
        double &rhoR = b.Q(i, j, k, 0);
        double betaR = rhoR / (2.0 * pR);
        double kR = uR * uR + vR * vR + wR * wR;

        double rhoBar = (rhoR + rhoL) / 2.0;
        double betaBar = (betaR + betaL) / 2.0;
        double uBar = (uR + uL) / 2.0;
        double vBar = (vR + vL) / 2.0;
        double wBar = (wR + wL) / 2.0;
        double kBar = (kR + kL) / 2.0;

        double &gamma = b.qh(i, j, k, 0);
        double U = uBar * b.jsx(i, j, k) + vBar * b.jsy(i, j, k) +
                   wBar * b.jsz(i, j, k);

        double eps = 0.01;
        // rhoHat
        double rhoHat;
        if (rhoL == rhoR) {
          double ksi = rhoL / rhoR;
          double f = (ksi - 1.0) / (ksi + 1.0);
          double fu = pow(f, 2.0);
          double F;
          if (fu < eps) {
            F = 1.0 + fu / 3.0 + fu * fu / 5.0 + fu * fu * fu / 7.0 +
                fu * fu * fu * fu / 9.0;
          } else {
            F = log(ksi) / 2.0 / fu;
          }
          rhoHat = (rhoR + rhoL) / (2 * F);
        } else {
          rhoHat = (rhoR - rhoL) / (log(rhoR) - log(rhoL));
        }
        // betaHat
        double betaHat;
        if (betaL == betaR) {
          double ksi = betaL / betaR;
          double f = (ksi - 1.0) / (ksi + 1.0);
          double fu = pow(f, 2.0);
          double F;
          if (fu < eps) {
            F = 1.0 + fu / 3.0 + fu * fu / 5.0 + fu * fu * fu / 7.0 +
                fu * fu * fu * fu / 9.0;
          } else {
            F = log(ksi) / 2.0 / fu;
          }
          betaHat = (betaR + betaL) / (2 * F);
        } else {
          betaHat = (betaR - betaL) / (log(betaR) - log(betaL));
        }

        double pTilde = rhoBar / (2.0 * betaBar);

        // Continuity rho*Ui
        b.jF(i, j, k, 0) = rhoHat * U;

        // x momentum rho*u*Ui+ p*Ax
        b.jF(i, j, k, 1) = b.jF(i, j, k, 0) * uBar + pTilde * b.jsx(i, j, k);

        // y momentum rho*v*Ui+ p*Ay
        b.jF(i, j, k, 2) = b.jF(i, j, k, 0) * vBar + pTilde * b.jsy(i, j, k);

        // w momentum rho*w*Ui+ p*Az
        b.jF(i, j, k, 3) = b.jF(i, j, k, 0) * wBar + pTilde * b.jsz(i, j, k);

        // Total energy (rhoE+ p)*Ui)
        b.jF(i, j, k, 4) =
            (1.0 / (2.0 * (gamma - 1.0) * betaHat) - 0.5 * kBar) *
                b.jF(i, j, k, 0) +
            uBar * b.jF(i, j, k, 1) + vBar * b.jF(i, j, k, 2) +
            wBar * b.jF(i, j, k, 3);
      });

  //-------------------------------------------------------------------------------------------|
  // k flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_k({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng});
  Kokkos::parallel_for(
      "2nd order KEEP k face conv fluxes", range_k,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double &pL = b.q(i, j, k - 1, 0);
        double &uL = b.q(i, j, k - 1, 1);
        double &vL = b.q(i, j, k - 1, 2);
        double &wL = b.q(i, j, k - 1, 3);
        double &rhoL = b.Q(i, j, k - 1, 0);
        double betaL = rhoL / (2.0 * pL);
        double kL = uL * uL + vL * vL + wL * wL;

        double &pR = b.q(i, j, k, 0);
        double &uR = b.q(i, j, k, 1);
        double &vR = b.q(i, j, k, 2);
        double &wR = b.q(i, j, k, 3);
        double &rhoR = b.Q(i, j, k, 0);
        double betaR = rhoR / (2.0 * pR);
        double kR = uR * uR + vR * vR + wR * wR;

        double rhoBar = (rhoR + rhoL) / 2.0;
        double betaBar = (betaR + betaL) / 2.0;
        double uBar = (uR + uL) / 2.0;
        double vBar = (vR + vL) / 2.0;
        double wBar = (wR + wL) / 2.0;
        double kBar = (kR + kL) / 2.0;

        double &gamma = b.qh(i, j, k, 0);
        double U = uBar * b.ksx(i, j, k) + vBar * b.ksy(i, j, k) +
                   wBar * b.ksz(i, j, k);

        double eps = 0.01;
        // rhoHat
        double rhoHat;
        if (rhoL == rhoR) {
          double ksi = rhoL / rhoR;
          double f = (ksi - 1.0) / (ksi + 1.0);
          double fu = pow(f, 2.0);
          double F;
          if (fu < eps) {
            F = 1.0 + fu / 3.0 + fu * fu / 5.0 + fu * fu * fu / 7.0 +
                fu * fu * fu * fu / 9.0;
          } else {
            F = log(ksi) / 2.0 / fu;
          }
          rhoHat = (rhoR + rhoL) / (2 * F);
        } else {
          rhoHat = (rhoR - rhoL) / (log(rhoR) - log(rhoL));
        }
        // betaHat
        double betaHat;
        if (betaL == betaR) {
          double ksi = betaL / betaR;
          double f = (ksi - 1.0) / (ksi + 1.0);
          double fu = pow(f, 2.0);
          double F;
          if (fu < eps) {
            F = 1.0 + fu / 3.0 + fu * fu / 5.0 + fu * fu * fu / 7.0 +
                fu * fu * fu * fu / 9.0;
          } else {
            F = log(ksi) / 2.0 / fu;
          }
          betaHat = (betaR + betaL) / (2 * F);
        } else {
          betaHat = (betaR - betaL) / (log(betaR) - log(betaL));
        }

        double pTilde = rhoBar / (2.0 * betaBar);

        // Continuity rho*Ui
        b.kF(i, j, k, 0) = rhoHat * U;

        // x momentum rho*u*Ui+ p*Ax
        b.kF(i, j, k, 1) = b.kF(i, j, k, 0) * uBar + pTilde * b.ksx(i, j, k);

        // y momentum rho*v*Ui+ p*Ay
        b.kF(i, j, k, 2) = b.kF(i, j, k, 0) * vBar + pTilde * b.ksy(i, j, k);

        // w momentum rho*w*Ui+ p*Az
        b.kF(i, j, k, 3) = b.kF(i, j, k, 0) * wBar + pTilde * b.ksz(i, j, k);

        // Total energy (rhoE+ p)*Ui)
        b.kF(i, j, k, 4) =
            (1.0 / (2.0 * (gamma - 1.0) * betaHat) - 0.5 * kBar) *
                b.kF(i, j, k, 0) +
            uBar * b.kF(i, j, k, 1) + vBar * b.kF(i, j, k, 2) +
            wBar * b.kF(i, j, k, 3);
      });
}
