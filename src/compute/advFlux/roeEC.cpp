#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"

void roeEC(block_ &b) {

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

        double &pR = b.q(i, j, k, 0);
        double &uR = b.q(i, j, k, 1);
        double &vR = b.q(i, j, k, 2);
        double &wR = b.q(i, j, k, 3);
        double &rhoR = b.Q(i, j, k, 0);

        double &gamma = b.qh(i, j, k, 0);

        double z1R = sqrt(rhoR / pR);
        double z2R = sqrt(rhoR / pR) * uR;
        double z3R = sqrt(rhoR / pR) * vR;
        double z4R = sqrt(rhoR / pR) * wR;
        double z5R = sqrt(rhoR / pR) * pR;

        double z1L = sqrt(rhoL / pL);
        double z2L = sqrt(rhoL / pL) * uL;
        double z3L = sqrt(rhoL / pL) * vL;
        double z4L = sqrt(rhoL / pL) * wL;
        double z5L = sqrt(rhoL / pL) * pL;

        // Compute fluxes
        double eps = 0.01;

        double z1Bar = 0.5 * (z1L + z1R);
        double z2Bar = 0.5 * (z2L + z2R);
        double z3Bar = 0.5 * (z3L + z3R);
        double z4Bar = 0.5 * (z4L + z4R);
        double z5Bar = 0.5 * (z5L + z5R);

        // z1Hat
        double z1Hat;
        if (z1L == z1R) {
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
          z1Hat = (z1R + z1L) / (2 * F);
        } else {
          z1Hat = (z1R - z1L) / (log(z1R) - log(z1L));
        }

        // z5Hat
        double z5Hat;
        if (z5L == z5R) {
          double ksi = z5L / z5R;
          double f = (ksi - 1.0) / (ksi + 1.0);
          double fu = pow(f, 2.0);
          double F;
          if (fu < eps) {
            F = 1.0 + fu / 3.0 + fu * fu / 5.0 + fu * fu * fu / 7.0 +
                fu * fu * fu * fu / 9.0;
          } else {
            F = log(ksi) / 2.0 / fu;
          }
          z5Hat = (z5R + z5L) / (2 * F);
        } else {
          z5Hat = (z5R - z5L) / (log(z5R) - log(z5L));
        }

        double rhoTilde = z1Bar * z5Hat;
        double uTilde = z2Bar / z1Bar;
        double vTilde = z3Bar / z1Bar;
        double wTilde = z4Bar / z1Bar;

        double UTilde = uTilde * b.isx(i, j, k) + vTilde * b.isy(i, j, k) +
                        wTilde * b.isz(i, j, k);

        double p1Tilde = z5Bar / z1Bar;
        double p2Tilde = (gamma + 1.0) / (2.0 * gamma) * z5Hat / z1Hat +
                         (gamma - 1.0) / (2.0 * gamma) * z5Bar / z1Bar;
        double aTilde = sqrt(gamma * p2Tilde / rhoTilde);
        double HTilde =
            pow(aTilde, 2.0) / (gamma - 1.0) +
            0.5 * (uTilde * uTilde + vTilde * vTilde + wTilde * wTilde);

        // Continuity rho*Ui
        b.iF(i, j, k, 0) = rhoTilde * UTilde;

        // x momentum rho*u*Ui+ p*Ax
        b.iF(i, j, k, 1) = b.iF(i, j, k, 0) * uTilde + p1Tilde * b.isx(i, j, k);

        // y momentum rho*v*Ui+ p*Ay
        b.iF(i, j, k, 2) = b.iF(i, j, k, 0) * vTilde + p1Tilde * b.isy(i, j, k);

        // w momentum rho*w*Ui+ p*Az
        b.iF(i, j, k, 3) = b.iF(i, j, k, 0) * wTilde + p1Tilde * b.isz(i, j, k);

        // Total energy (rhoE+ p)*Ui)
        b.iF(i, j, k, 4) = b.iF(i, j, k, 0) * HTilde;
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

        double &pR = b.q(i, j, k, 0);
        double &uR = b.q(i, j, k, 1);
        double &vR = b.q(i, j, k, 2);
        double &wR = b.q(i, j, k, 3);
        double &rhoR = b.Q(i, j, k, 0);

        double &gamma = b.qh(i, j, k, 0);

        double z1R = sqrt(rhoR / pR);
        double z2R = sqrt(rhoR / pR) * uR;
        double z3R = sqrt(rhoR / pR) * vR;
        double z4R = sqrt(rhoR / pR) * wR;
        double z5R = sqrt(rhoR / pR) * pR;

        double z1L = sqrt(rhoL / pL);
        double z2L = sqrt(rhoL / pL) * uL;
        double z3L = sqrt(rhoL / pL) * vL;
        double z4L = sqrt(rhoL / pL) * wL;
        double z5L = sqrt(rhoL / pL) * pL;

        // Compute fluxes
        double eps = 0.01;

        double z1Bar = 0.5 * (z1L + z1R);
        double z2Bar = 0.5 * (z2L + z2R);
        double z3Bar = 0.5 * (z3L + z3R);
        double z4Bar = 0.5 * (z4L + z4R);
        double z5Bar = 0.5 * (z5L + z5R);

        // z1Hat
        double z1Hat;
        if (z1L == z1R) {
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
          z1Hat = (z1R + z1L) / (2 * F);
        } else {
          z1Hat = (z1R - z1L) / (log(z1R) - log(z1L));
        }

        // z5Hat
        double z5Hat;
        if (z5L == z5R) {
          double ksi = z5L / z5R;
          double f = (ksi - 1.0) / (ksi + 1.0);
          double fu = pow(f, 2.0);
          double F;
          if (fu < eps) {
            F = 1.0 + fu / 3.0 + fu * fu / 5.0 + fu * fu * fu / 7.0 +
                fu * fu * fu * fu / 9.0;
          } else {
            F = log(ksi) / 2.0 / fu;
          }
          z5Hat = (z5R + z5L) / (2 * F);
        } else {
          z5Hat = (z5R - z5L) / (log(z5R) - log(z5L));
        }

        double rhoTilde = z1Bar * z5Hat;
        double uTilde = z2Bar / z1Bar;
        double vTilde = z3Bar / z1Bar;
        double wTilde = z4Bar / z1Bar;

        double UTilde = uTilde * b.jsx(i, j, k) + vTilde * b.jsy(i, j, k) +
                        wTilde * b.jsz(i, j, k);

        double p1Tilde = z5Bar / z1Bar;
        double p2Tilde = (gamma + 1.0) / (2.0 * gamma) * z5Hat / z1Hat +
                         (gamma - 1.0) / (2.0 * gamma) * z5Bar / z1Bar;
        double aTilde = sqrt(gamma * p2Tilde / rhoTilde);
        double HTilde =
            pow(aTilde, 2.0) / (gamma - 1.0) +
            0.5 * (uTilde * uTilde + vTilde * vTilde + wTilde * wTilde);

        // Continuity rho*Ui
        b.jF(i, j, k, 0) = rhoTilde * UTilde;

        // x momentum rho*u*Ui+ p*Ax
        b.jF(i, j, k, 1) = b.jF(i, j, k, 0) * uTilde + p1Tilde * b.jsx(i, j, k);

        // y momentum rho*v*Ui+ p*Ay
        b.jF(i, j, k, 2) = b.jF(i, j, k, 0) * vTilde + p1Tilde * b.jsy(i, j, k);

        // w momentum rho*w*Ui+ p*Az
        b.jF(i, j, k, 3) = b.jF(i, j, k, 0) * wTilde + p1Tilde * b.jsz(i, j, k);

        // Total energy (rhoE+ p)*Ui)
        b.jF(i, j, k, 4) = b.jF(i, j, k, 0) * HTilde;
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

        double &pR = b.q(i, j, k, 0);
        double &uR = b.q(i, j, k, 1);
        double &vR = b.q(i, j, k, 2);
        double &wR = b.q(i, j, k, 3);
        double &rhoR = b.Q(i, j, k, 0);

        double &gamma = b.qh(i, j, k, 0);

        double z1R = sqrt(rhoR / pR);
        double z2R = sqrt(rhoR / pR) * uR;
        double z3R = sqrt(rhoR / pR) * vR;
        double z4R = sqrt(rhoR / pR) * wR;
        double z5R = sqrt(rhoR / pR) * pR;

        double z1L = sqrt(rhoL / pL);
        double z2L = sqrt(rhoL / pL) * uL;
        double z3L = sqrt(rhoL / pL) * vL;
        double z4L = sqrt(rhoL / pL) * wL;
        double z5L = sqrt(rhoL / pL) * pL;

        // Compute fluxes
        double eps = 0.01;

        double z1Bar = 0.5 * (z1L + z1R);
        double z2Bar = 0.5 * (z2L + z2R);
        double z3Bar = 0.5 * (z3L + z3R);
        double z4Bar = 0.5 * (z4L + z4R);
        double z5Bar = 0.5 * (z5L + z5R);

        // z1Hat
        double z1Hat;
        if (z1L == z1R) {
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
          z1Hat = (z1R + z1L) / (2 * F);
        } else {
          z1Hat = (z1R - z1L) / (log(z1R) - log(z1L));
        }

        // z5Hat
        double z5Hat;
        if (z5L == z5R) {
          double ksi = z5L / z5R;
          double f = (ksi - 1.0) / (ksi + 1.0);
          double fu = pow(f, 2.0);
          double F;
          if (fu < eps) {
            F = 1.0 + fu / 3.0 + fu * fu / 5.0 + fu * fu * fu / 7.0 +
                fu * fu * fu * fu / 9.0;
          } else {
            F = log(ksi) / 2.0 / fu;
          }
          z5Hat = (z5R + z5L) / (2 * F);
        } else {
          z5Hat = (z5R - z5L) / (log(z5R) - log(z5L));
        }

        double rhoTilde = z1Bar * z5Hat;
        double uTilde = z2Bar / z1Bar;
        double vTilde = z3Bar / z1Bar;
        double wTilde = z4Bar / z1Bar;

        double UTilde = uTilde * b.ksx(i, j, k) + vTilde * b.ksy(i, j, k) +
                        wTilde * b.ksz(i, j, k);

        double p1Tilde = z5Bar / z1Bar;
        double p2Tilde = (gamma + 1.0) / (2.0 * gamma) * z5Hat / z1Hat +
                         (gamma - 1.0) / (2.0 * gamma) * z5Bar / z1Bar;
        double aTilde = sqrt(gamma * p2Tilde / rhoTilde);
        double HTilde =
            pow(aTilde, 2.0) / (gamma - 1.0) +
            0.5 * (uTilde * uTilde + vTilde * vTilde + wTilde * wTilde);

        // Continuity rho*Ui
        b.kF(i, j, k, 0) = rhoTilde * UTilde;

        // x momentum rho*u*Ui+ p*Ax
        b.kF(i, j, k, 1) = b.kF(i, j, k, 0) * uTilde + p1Tilde * b.ksx(i, j, k);

        // y momentum rho*v*Ui+ p*Ay
        b.kF(i, j, k, 2) = b.kF(i, j, k, 0) * vTilde + p1Tilde * b.ksy(i, j, k);

        // w momentum rho*w*Ui+ p*Az
        b.kF(i, j, k, 3) = b.kF(i, j, k, 0) * wTilde + p1Tilde * b.ksz(i, j, k);

        // Total energy (rhoE+ p)*Ui)
        b.kF(i, j, k, 4) = b.kF(i, j, k, 0) * HTilde;
      });
}
