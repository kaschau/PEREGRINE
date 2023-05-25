#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "math.h"

void applyFlux(block_ &b, double[]) {

  //-------------------------------------------------------------------------------------------|
  // Apply fluxes to cc range
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "Apply current fluxes to RHS", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        // Add fluxes to RHS
        b.dQ(i, j, k, l) +=
            (b.iF(i, j, k, l) + b.jF(i, j, k, l) + b.kF(i, j, k, l)) /
            b.J(i, j, k);

        b.dQ(i, j, k, l) -= (b.iF(i + 1, j, k, l) + b.jF(i, j + 1, k, l) +
                             b.kF(i, j, k + 1, l)) /
                            b.J(i, j, k);
      });
  MDRange3 range_cc3({b.ng, b.ng, b.ng},
                     {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "Apply current fluxes to RHS", range_cc3,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Add fluxes to RHS
        double cv = b.qh(i, j, k, 1) / b.qh(i, j, k, 0);
        double &T = b.q(i, j, k, 4);
        double &rho = b.Q(i, j, k, 0);
        double R = b.qh(i, j, k, 1) - cv;
        // THIS IS CRITICAL
        // s MUST be derived, not used from the variable b.s/b.Q in order
        // for us to get zero entropy generation in entropy from this
        // evolution equation.
        double s = cv * log(T) - R * log(rho);
        double &u = b.q(i, j, k, 1);
        double &v = b.q(i, j, k, 2);
        double &w = b.q(i, j, k, 3);
        double h = b.qh(i, j, k, 2) / b.Q(i, j, k, 0);
        double v0 = s + (-h + 0.5 * (pow(u, 2) + pow(v, 2) + pow(w, 2))) / T;
        double v1 = -u / T;
        double v2 = -v / T;
        double v3 = -w / T;
        double v4 = 1.0 / T;
        b.ds(i, j, k) +=
            ((b.iF(i, j, k, 0) + b.jF(i, j, k, 0) + b.kF(i, j, k, 0)) * v0 +
             (b.iF(i, j, k, 1) + b.jF(i, j, k, 1) + b.kF(i, j, k, 1)) * v1 +
             (b.iF(i, j, k, 2) + b.jF(i, j, k, 2) + b.kF(i, j, k, 2)) * v2 +
             (b.iF(i, j, k, 3) + b.jF(i, j, k, 3) + b.kF(i, j, k, 3)) * v3 +
             (b.iF(i, j, k, 4) + b.jF(i, j, k, 4) + b.kF(i, j, k, 4)) * v4) /
            b.J(i, j, k);

        b.ds(i, j, k) -= ((b.iF(i + 1, j, k, 0) + b.jF(i, j + 1, k, 0) +
                           b.kF(i, j, k + 1, 0)) *
                              v0 +
                          (b.iF(i + 1, j, k, 1) + b.jF(i, j + 1, k, 1) +
                           b.kF(i, j, k + 1, 1)) *
                              v1 +
                          (b.iF(i + 1, j, k, 2) + b.jF(i, j + 1, k, 2) +
                           b.kF(i, j, k + 1, 2)) *
                              v2 +
                          (b.iF(i + 1, j, k, 3) + b.jF(i, j + 1, k, 3) +
                           b.kF(i, j, k + 1, 3)) *
                              v3 +
                          (b.iF(i + 1, j, k, 4) + b.jF(i, j + 1, k, 4) +
                           b.kF(i, j, k + 1, 4)) *
                              v4) /
                         b.J(i, j, k);
      });
}

void applyHybridFlux(block_ &b, const double &primary) {

  //-------------------------------------------------------------------------------------------|
  // Apply fluxes to cc range
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "Apply hybrid fluxes to RHS", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        // Compute switch on face
        double iFphi = fmax(b.phi(i, j, k, 0), b.phi(i - 1, j, k, 0));
        double iFphi1 = fmax(b.phi(i, j, k, 0), b.phi(i + 1, j, k, 0));
        double jFphi = fmax(b.phi(i, j, k, 1), b.phi(i, j - 1, k, 1));
        double jFphi1 = fmax(b.phi(i, j, k, 1), b.phi(i, j + 1, k, 1));
        double kFphi = fmax(b.phi(i, j, k, 2), b.phi(i, j, k - 1, 2));
        double kFphi1 = fmax(b.phi(i, j, k, 2), b.phi(i, j, k + 1, 2));

        double dPrimary = 2.0 * primary - 1.0;

        // Add fluxes to RHS
        // format is F_primary*(1-switch) + F_secondary*(switch)
        // so when switch == 0, we dont switch from primary
        //    when switch == 1 we completely switch
        b.dQ(i, j, k, l) += (b.iF(i, j, k, l) * (primary - iFphi * dPrimary) +
                             b.jF(i, j, k, l) * (primary - jFphi * dPrimary) +
                             b.kF(i, j, k, l) * (primary - kFphi * dPrimary)) /
                            b.J(i, j, k);

        b.dQ(i, j, k, l) -=
            (b.iF(i + 1, j, k, l) * (primary - iFphi1 * dPrimary) +
             b.jF(i, j + 1, k, l) * (primary - jFphi1 * dPrimary) +
             b.kF(i, j, k + 1, l) * (primary - kFphi1 * dPrimary)) /
            b.J(i, j, k);
      });
}

void applyDissipationFlux(block_ &b, double[]) {

  //-------------------------------------------------------------------------------------------|
  // Apply fluxes to cc range
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "Apply dissipaiton fluxes to RHS", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        // Add fluxes to RHS
        b.dQ(i, j, k, l) -=
            (b.iF(i, j, k, l) + b.jF(i, j, k, l) + b.kF(i, j, k, l)) /
            b.J(i, j, k);

        b.dQ(i, j, k, l) += (b.iF(i + 1, j, k, l) + b.jF(i, j + 1, k, l) +
                             b.kF(i, j, k + 1, l)) /
                            b.J(i, j, k);
      });
  MDRange3 range_cc3({b.ng, b.ng, b.ng},
                     {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "Apply dissipaiton fluxes to RHS", range_cc3,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Add fluxes to RHS
        double s = b.s(i, j, k) / b.Q(i, j, k, 0);
        double &u = b.q(i, j, k, 1);
        double &v = b.q(i, j, k, 2);
        double &w = b.q(i, j, k, 3);
        double h = b.qh(i, j, k, 2) / b.Q(i, j, k, 0);
        double &T = b.q(i, j, k, 4);
        double v0 = s + (-h + 0.5 * (pow(u, 2) + pow(v, 2) + pow(w, 2))) / T;
        double v1 = -u / T;
        double v2 = -v / T;
        double v3 = -w / T;
        double v4 = 1.0 / T;
        b.ds(i, j, k) -=
            ((b.iF(i, j, k, 0) + b.jF(i, j, k, 0) + b.kF(i, j, k, 0)) * v0 +
             (b.iF(i, j, k, 1) + b.jF(i, j, k, 1) + b.kF(i, j, k, 1)) * v1 +
             (b.iF(i, j, k, 2) + b.jF(i, j, k, 2) + b.kF(i, j, k, 2)) * v2 +
             (b.iF(i, j, k, 3) + b.jF(i, j, k, 3) + b.kF(i, j, k, 3)) * v3 +
             (b.iF(i, j, k, 4) + b.jF(i, j, k, 4) + b.kF(i, j, k, 4)) * v4) /
            b.J(i, j, k);

        b.ds(i, j, k) += ((b.iF(i + 1, j, k, 0) + b.jF(i, j + 1, k, 0) +
                           b.kF(i, j, k + 1, 0)) *
                              v0 +
                          (b.iF(i + 1, j, k, 1) + b.jF(i, j + 1, k, 1) +
                           b.kF(i, j, k + 1, 1)) *
                              v1 +
                          (b.iF(i + 1, j, k, 2) + b.jF(i, j + 1, k, 2) +
                           b.kF(i, j, k + 1, 2)) *
                              v2 +
                          (b.iF(i + 1, j, k, 3) + b.jF(i, j + 1, k, 3) +
                           b.kF(i, j, k + 1, 3)) *
                              v3 +
                          (b.iF(i + 1, j, k, 4) + b.jF(i, j + 1, k, 4) +
                           b.kF(i, j, k + 1, 4)) *
                              v4) /
                         b.J(i, j, k);
      });
}
