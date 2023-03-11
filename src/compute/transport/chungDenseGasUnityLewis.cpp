#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <math.h>

// References
//
// Generalized Multiparameter Correlation for Nonpolar and Polar Fluid Transport
// Properties
//     Ting-Horng Chung
//     Ajlan
//     Lee
//     Starling
//     Ind. Eng. Chem. Res. 1988, 27,671-679
//
// The Properties of Gases and Liquids
//     Bruce Poling
//     Prausnitz
//     O'Connell
//     5th Edition, 2001

void chungDenseGasUnityLewis(block_ &b, const thtrdat_ &th, const int &nface,
                             const int &indxI /*=0*/, const int &indxJ /*=0*/,
                             const int &indxK /*=0*/) {
#ifndef NSCOMPILE
  Kokkos::Experimental::UniqueToken<execSpace> token;
  int numIds = token.size();
  const int ns = th.ns;
  twoDview Y("Y", numIds, ns);
  twoDview X("X", numIds, ns);
  twoDview mu_sp("mu_sp", numIds, ns);
  twoDview kappa_sp("kappa_sp", numIds, ns);
#endif

#ifdef NSCOMPILE
#define Y(INDEX) Y[INDEX]
#define X(INDEX) X[INDEX]
#define mu_sp(INDEX) mu_sp[INDEX]
#define kappa_sp(INDEX) kappa_sp[INDEX]
#define ns NS
#else
#define Y(INDEX) Y(id, INDEX)
#define X(INDEX) X(id, INDEX)
#define mu_sp(INDEX) mu_sp(id, INDEX)
#define kappa_sp(INDEX) kappa_sp(id, INDEX)
#endif

  MDRange3 range = getRange3(b, nface, indxI, indxJ, indxK);
  Kokkos::parallel_for(
      "Chung trans props unity Lewis", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
#ifndef NSCOMPILE
        int id = token.acquire();
#endif

        double &T = b.q(i, j, k, 4);
#ifdef NSCOMPILE
        double Y(ns);
        double X(ns);
        double mu_sp(ns);
        double kappa_sp(ns);
#endif

        // Compute nth species Y
        Y(ns - 1) = 1.0;
        for (int n = 0; n < ns - 1; n++) {
          Y(n) = b.q(i, j, k, 5 + n);
          Y(ns - 1) -= Y(n);
        }
        Y(ns - 1) = fmax(0.0, Y(ns - 1));

        // Update mixture properties
        // Mole fractions
        double MWmix = 0.0;
        {
          double mass = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            mass += Y(n) / th.MW(n);
          }
          // Mean molecular weight, mole fraction
          for (int n = 0; n <= ns - 1; n++) {
            X(n) = Y(n) / th.MW(n) / mass;
            MWmix += X(n) * th.MW(n);
          }
        }

        for (int n = 0; n <= ns - 1; n++) {
          const double Acoeff = 1.16145;
          const double Bcoeff = 0.14874;
          const double Ccoeff = 0.52487;
          const double Dcoeff = 0.77320;
          const double Ecoeff = 2.16178;
          const double Fcoeff = 2.43787;

          // In the Chung paper, Vcrit is in cm^3/mol, but we store Vcrit in
          // m^3/kg
          double Vc = th.Vcrit(n) * th.MW(n) * 1.0e3;

          // Dense gas viscosity
          double Tr = T / th.Tcrit(n);
          double Tstar = 1.2593 * Tr;
          double Omegav = Acoeff * pow(Tstar, -Bcoeff) +
                          Ccoeff * exp(-Dcoeff * Tstar) +
                          Ecoeff * exp(-Fcoeff * Tstar);

          double Fc = 1.0 - 0.2756 * th.acentric(n) +
                      0.059035 * pow(th.redDipole(n), 4.0); // + kij??

          // Viscosity for dense fluids
          double &A1 = th.chungA(n, 0);
          double &A2 = th.chungA(n, 1);
          double &A3 = th.chungA(n, 2);
          double &A4 = th.chungA(n, 3);
          double &A5 = th.chungA(n, 4);
          double &A6 = th.chungA(n, 5);
          double &A7 = th.chungA(n, 6);
          double &A8 = th.chungA(n, 7);
          double &A9 = th.chungA(n, 8);
          double &A10 = th.chungA(n, 9);

          double rhocm = b.Q(i, j, k, 0) / th.MW(n) * 1e-3;
          double Yy = rhocm * Vc / 6.0;
          double G1 = (1.0 - 0.5 * Yy) / pow(1.0 - Yy, 3.0);
          double G2 = (A1 * (1.0 - exp(-A4 * Yy)) / Yy +
                       A2 * G1 * exp(A5 * Yy) + A3 * G1) /
                      (A1 * A4 + A2 + A3);

          double etaStarStar = A7 * pow(Yy, 2.0) * G2 *
                               exp(A8 + A9 / Tstar + A10 * pow(Tstar, -2.0));
          double etaStar =
              sqrt(Tstar) / Omegav * (Fc / G2 + A6 * Yy) + etaStarStar;

          // Compute final viscosity, convert to SI units
          mu_sp(n) = etaStar * 36.344 * sqrt(th.MW(n) * th.Tcrit(n)) /
                     pow(Vc, 2.0 / 3.0) * 1e-7;

          // Dilute gas thermal conductivity
          double alpha = b.qh(i, j, k, 1) * 0.001 * th.MW(n) /
                             b.qh(i, j, k, 0) / (th.Ru / 1000.0) -
                         1.5;
          double beta = 0.7862 - 0.7109 * th.acentric(n) +
                        1.3168 * pow(th.acentric(n), 2.0);
          double eta0 = 4.0785e-5 * sqrt(th.MW(n) * T) /
                        (pow(Vc, 2.0 / 3.0) * Omegav) * Fc;
          double Z = 2.0 + 10.5 * pow(Tr, 2.0);
          double Psi =
              1.0 +
              alpha * ((0.215 + 0.28288 * alpha - 1.061 * beta + 0.26665 * Z) /
                       (0.6366 + beta * Z + 1.061 * alpha * beta));
          double lambda0 = 7.452 * eta0 / th.MW(n) * Psi;

          // Dilute thermal conductivity, in cal/(cm.s.K) so need to convert
          double &B1 = th.chungB(n, 0);
          double &B2 = th.chungB(n, 1);
          double &B3 = th.chungB(n, 2);
          double &B4 = th.chungB(n, 3);
          double &B5 = th.chungB(n, 4);
          double &B6 = th.chungB(n, 5);
          double &B7 = th.chungB(n, 6);

          double H2 = (B1 * (1.0 - exp(-B4 * Yy)) / Yy +
                       B2 * G1 * exp(B5 * Yy) + B3 * G1) /
                      (B1 * B4 + B2 + B3);

          double lambdak = lambda0 * (1.0 / H2 + B6 * Yy);
          double lambdap =
              (3.039e-4 * sqrt(th.Tcrit(n) / th.MW(n)) / pow(Vc, 2.0 / 3.0)) *
              B7 * pow(Yy, 2.0) * H2 * sqrt(Tr);

          // Compute final thermal conductivity, convert to SI units
          kappa_sp(n) = (lambdak + lambdap) * 418.68;
        }

        // Now every species' property is computed, generate mixture values

        // viscosity mixture
        double mu = 0.0;
        for (int n = 0; n <= ns - 1; n++) {
          double phitemp = 0.0;
          for (int n2 = 0; n2 <= ns - 1; n2++) {
            double phi = pow((1.0 + sqrt(mu_sp(n) / mu_sp(n2) *
                                         sqrt(th.MW(n2) / th.MW(n)))),
                             2.0) /
                         (sqrt(8.0) * sqrt(1 + th.MW(n) / th.MW(n2)));
            phitemp += phi * X(n2);
          }
          mu += mu_sp(n) * X(n) / phitemp;
        }

        // thermal conductivity mixture
        double kappa = 0.0;
        {
          double sum1 = 0.0;
          double sum2 = 0.0;
          for (int n = 0; n <= ns - 1; n++) {
            sum1 += X(n) * kappa_sp(n);
            sum2 += X(n) / kappa_sp(n);
          }
          kappa = 0.5 * (sum1 + 1.0 / sum2);
        }

        // Set values of new properties
        // viscocity
        b.qt(i, j, k, 0) = mu;
        // thermal conductivity
        b.qt(i, j, k, 1) = kappa;
        // NOTE: Unity Lewis number approximation!
        for (int n = 0; n <= ns - 1; n++) {
          b.qt(i, j, k, 2 + n) = kappa / (b.Q(i, j, k, 0) * b.qh(i, j, k, 1));
        }

#ifndef NSCOMPILE
        token.release(id);
#endif
      });

  // if ns == 1, we need the diffusion coeff to be zero
  // so the viscous flux correction term is zero in
  // diffusiveFlux.cpp
  if (ns == 1) {
    MDRange3 range = getRange3(b, nface, indxI, indxJ, indxK);
    Kokkos::parallel_for(
        "Const Props Transport", range,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          b.qt(i, j, k, 2) = 0.0;
        });
  }
}
