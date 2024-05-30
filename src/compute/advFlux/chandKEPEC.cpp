#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"

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
        double &pL = b.q(i - iMod, j - jMod, k - kMod, 0);
        double &uL = b.q(i - iMod, j - jMod, k - kMod, 1);
        double &vL = b.q(i - iMod, j - jMod, k - kMod, 2);
        double &wL = b.q(i - iMod, j - jMod, k - kMod, 3);
        double &rhoL = b.Q(i - iMod, j - jMod, k - kMod, 0);
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
        double U =
            uBar * isx(i, j, k) + vBar * isy(i, j, k) + wBar * isz(i, j, k);

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
        iF(i, j, k, 0) = rhoHat * U;

        // x momentum rho*u*Ui+ p*Ax
        iF(i, j, k, 1) = iF(i, j, k, 0) * uBar + pTilde * isx(i, j, k);

        // y momentum rho*v*Ui+ p*Ay
        iF(i, j, k, 2) = iF(i, j, k, 0) * vBar + pTilde * isy(i, j, k);

        // w momentum rho*w*Ui+ p*Az
        iF(i, j, k, 3) = iF(i, j, k, 0) * wBar + pTilde * isz(i, j, k);

        // Total energy (rhoE+ p)*Ui)

        iF(i, j, k, 4) = (1.0 / (2.0 * (gamma - 1.0) * betaHat) - 0.5 * kBar) *
                             iF(i, j, k, 0) +
                         uBar * iF(i, j, k, 1) + vBar * iF(i, j, k, 2) +
                         wBar * iF(i, j, k, 3);
      });
}

void chandKEPEC(block_ &b) {
  computeFlux(b, b.iF, b.isx, b.isy, b.isz, 1, 0, 0);
  computeFlux(b, b.jF, b.jsx, b.jsy, b.jsz, 0, 1, 0);
  computeFlux(b, b.kF, b.ksx, b.ksy, b.ksz, 0, 0, 1);
}
