#include "conserved.hpp"
#include "diffusion.hpp"
#include "kernel.hpp"
#include "mixing.hpp"
#include "mixingRule.hpp"

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

PG_RANGE(cellCenters)
struct chungDenseGas {
  cellCenterIn Q, q, qh;
  cellCenterOut qt;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const fpdtype &T = q(1);
    fpdtype X[ns];
    fpdtype mu_sp[ns];
    fpdtype kappa_sp[ns];

    // the mole fractions off the conserved state, the last mass fraction
    // kept non-negative
    const fpdtype rhoinv = 1.0 / Q(0);
    const auto Yof = massFractions(Q, rhoinv);
    const auto Y = [&](const int n) {
      return n == ns - 1 ? fmax(0.0, Yof(n)) : Yof(n);
    };
    const fpdtype MWmix = moleFractions(Y, X);

    for (int n = 0; n <= ns - 1; n++) {
      const fpdtype Acoeff = 1.16145;
      const fpdtype Bcoeff = 0.14874;
      const fpdtype Ccoeff = 0.52487;
      const fpdtype Dcoeff = 0.77320;
      const fpdtype Ecoeff = 2.16178;
      const fpdtype Fcoeff = 2.43787;

      // In the Chung paper, Vcrit is in cm^3/mol, but we store Vcrit
      // in m^3/kg
      fpdtype Vc = Vcrit(n) * MW(n) * 1.0e3;

      // Dense gas viscosity
      fpdtype Tr = T / Tcrit(n);
      fpdtype Tstar = 1.2593 * Tr;
      fpdtype Omegav = Acoeff * pow(Tstar, -Bcoeff) +
                       Ccoeff * exp(-Dcoeff * Tstar) +
                       Ecoeff * exp(-Fcoeff * Tstar);

      fpdtype Fc = 1.0 - 0.2756 * acentric(n) +
                   0.059035 * pow(redDipole(n), 4.0); // + kij??

      // Viscosity for dense fluids
      const fpdtype &A1 = chungA(n, 0);
      const fpdtype &A2 = chungA(n, 1);
      const fpdtype &A3 = chungA(n, 2);
      const fpdtype &A4 = chungA(n, 3);
      const fpdtype &A5 = chungA(n, 4);
      const fpdtype &A6 = chungA(n, 5);
      const fpdtype &A7 = chungA(n, 6);
      const fpdtype &A8 = chungA(n, 7);
      const fpdtype &A9 = chungA(n, 8);
      const fpdtype &A10 = chungA(n, 9);

      fpdtype rhocm = Q(0) * MWinv(n) * 1e-3;
      fpdtype Yy = rhocm * Vc / 6.0;
      fpdtype G1 = (1.0 - 0.5 * Yy) / pow(1.0 - Yy, 3.0);
      fpdtype G2 =
          (A1 * (1.0 - exp(-A4 * Yy)) / Yy + A2 * G1 * exp(A5 * Yy) + A3 * G1) /
          (A1 * A4 + A2 + A3);

      fpdtype etaStarStar = A7 * pow(Yy, 2.0) * G2 *
                            exp(A8 + A9 / Tstar + A10 * pow(Tstar, -2.0));
      fpdtype etaStar =
          sqrt(Tstar) / Omegav * (Fc / G2 + A6 * Yy) + etaStarStar;

      // Compute final viscosity, convert to SI units
      mu_sp[n] =
          etaStar * 36.344 * sqrt(MW(n) * Tcrit(n)) / pow(Vc, 2.0 / 3.0) * 1e-7;

      // Dilute gas thermal conductivity
      fpdtype alpha = qh(1) * 0.001 * MW(n) / qh(0) / (Ru / 1000.0) - 1.5;
      fpdtype beta =
          0.7862 - 0.7109 * acentric(n) + 1.3168 * pow(acentric(n), 2.0);
      fpdtype eta0 =
          4.0785e-5 * sqrt(MW(n) * T) / (pow(Vc, 2.0 / 3.0) * Omegav) * Fc;
      fpdtype Z = 2.0 + 10.5 * pow(Tr, 2.0);
      fpdtype Psi =
          1.0 +
          alpha * ((0.215 + 0.28288 * alpha - 1.061 * beta + 0.26665 * Z) /
                   (0.6366 + beta * Z + 1.061 * alpha * beta));
      fpdtype lambda0 = 7.452 * eta0 * MWinv(n) * Psi;

      // Dilute thermal conductivity, in cal/(cm.s.K) so need to
      // convert
      const fpdtype &B1 = chungB(n, 0);
      const fpdtype &B2 = chungB(n, 1);
      const fpdtype &B3 = chungB(n, 2);
      const fpdtype &B4 = chungB(n, 3);
      const fpdtype &B5 = chungB(n, 4);
      const fpdtype &B6 = chungB(n, 5);
      const fpdtype &B7 = chungB(n, 6);

      fpdtype H2 =
          (B1 * (1.0 - exp(-B4 * Yy)) / Yy + B2 * G1 * exp(B5 * Yy) + B3 * G1) /
          (B1 * B4 + B2 + B3);

      fpdtype lambdak = lambda0 * (1.0 / H2 + B6 * Yy);
      fpdtype lambdap =
          (3.039e-4 * sqrt(Tcrit(n) * MWinv(n)) / pow(Vc, 2.0 / 3.0)) * B7 *
          pow(Yy, 2.0) * H2 * sqrt(Tr);

      // Compute final thermal conductivity, convert to SI units
      kappa_sp[n] = (lambdak + lambdap) * 418.68;
      // Wilke's rule wants sqrt(mu)
      mu_sp[n] = sqrt(mu_sp[n]);
    }

    // the mixture: Wilke's rule, the series-parallel mean
    qt(0) = mixingRule::viscosity(X, mu_sp);
    const fpdtype kappa = mixtureConductivity(X, kappa_sp);
    qt(1) = kappa;
    // the species diffusion coefficients from the case's diffusion model
    fpdtype D[ns];
    diffusion::coefficients({q(0), T, log(T), rhoinv, qh(1), kappa, MWmix, X},
                            D);
    for (int n = 0; n <= ns - 1; n++) {
      qt(2 + n) = D[n];
    }
  }
};

PG_ABI void pgChungDenseGas(const chungDenseGas &k, const pgTiling &t) {
  forCells("Chung dense gas trans props", t, k);
}
