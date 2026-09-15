#include "kernel.hpp"

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

PG_RANGE(cellCenters, halo = ng)
struct chungDenseGasUnityLewis {
  cellCenterIn Q, q, qh;
  cellCenterOut qt;
  record MW, Tcrit, Vcrit, acentric, chungA, chungB, lewis, redDipole;
  double Ru;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const double &T = q(4);
    double Y[ns];
    double X[ns];
    double mu_sp[ns];
    double kappa_sp[ns];

    // Compute nth species Y
    Y[ns - 1] = 1.0;
    for (int n = 0; n < ns - 1; n++) {
      Y[n] = q(5 + n);
      Y[ns - 1] -= Y[n];
    }
    Y[ns - 1] = fmax(0.0, Y[ns - 1]);

    // Update mixture properties
    // Mole fractions
    {
      double mass = 0.0;
      for (int n = 0; n <= ns - 1; n++) {
        mass += Y[n] / MW(n);
      }
      // Mean molecular weight, mole fraction
      for (int n = 0; n <= ns - 1; n++) {
        X[n] = Y[n] / MW(n) / mass;
      }
    }

    for (int n = 0; n <= ns - 1; n++) {
      const double Acoeff = 1.16145;
      const double Bcoeff = 0.14874;
      const double Ccoeff = 0.52487;
      const double Dcoeff = 0.77320;
      const double Ecoeff = 2.16178;
      const double Fcoeff = 2.43787;

      // In the Chung paper, Vcrit is in cm^3/mol, but we store Vcrit
      // in m^3/kg
      double Vc = Vcrit(n) * MW(n) * 1.0e3;

      // Dense gas viscosity
      double Tr = T / Tcrit(n);
      double Tstar = 1.2593 * Tr;
      double Omegav = Acoeff * pow(Tstar, -Bcoeff) +
                      Ccoeff * exp(-Dcoeff * Tstar) +
                      Ecoeff * exp(-Fcoeff * Tstar);

      double Fc = 1.0 - 0.2756 * acentric(n) +
                  0.059035 * pow(redDipole(n), 4.0); // + kij??

      // Viscosity for dense fluids
      const double &A1 = chungA(n, 0);
      const double &A2 = chungA(n, 1);
      const double &A3 = chungA(n, 2);
      const double &A4 = chungA(n, 3);
      const double &A5 = chungA(n, 4);
      const double &A6 = chungA(n, 5);
      const double &A7 = chungA(n, 6);
      const double &A8 = chungA(n, 7);
      const double &A9 = chungA(n, 8);
      const double &A10 = chungA(n, 9);

      double rhocm = Q(0) / MW(n) * 1e-3;
      double Yy = rhocm * Vc / 6.0;
      double G1 = (1.0 - 0.5 * Yy) / pow(1.0 - Yy, 3.0);
      double G2 =
          (A1 * (1.0 - exp(-A4 * Yy)) / Yy + A2 * G1 * exp(A5 * Yy) + A3 * G1) /
          (A1 * A4 + A2 + A3);

      double etaStarStar = A7 * pow(Yy, 2.0) * G2 *
                           exp(A8 + A9 / Tstar + A10 * pow(Tstar, -2.0));
      double etaStar = sqrt(Tstar) / Omegav * (Fc / G2 + A6 * Yy) + etaStarStar;

      // Compute final viscosity, convert to SI units
      mu_sp[n] =
          etaStar * 36.344 * sqrt(MW(n) * Tcrit(n)) / pow(Vc, 2.0 / 3.0) * 1e-7;

      // Dilute gas thermal conductivity
      double alpha = qh(1) * 0.001 * MW(n) / qh(0) / (Ru / 1000.0) - 1.5;
      double beta =
          0.7862 - 0.7109 * acentric(n) + 1.3168 * pow(acentric(n), 2.0);
      double eta0 =
          4.0785e-5 * sqrt(MW(n) * T) / (pow(Vc, 2.0 / 3.0) * Omegav) * Fc;
      double Z = 2.0 + 10.5 * pow(Tr, 2.0);
      double Psi =
          1.0 +
          alpha * ((0.215 + 0.28288 * alpha - 1.061 * beta + 0.26665 * Z) /
                   (0.6366 + beta * Z + 1.061 * alpha * beta));
      double lambda0 = 7.452 * eta0 / MW(n) * Psi;

      // Dilute thermal conductivity, in cal/(cm.s.K) so need to
      // convert
      const double &B1 = chungB(n, 0);
      const double &B2 = chungB(n, 1);
      const double &B3 = chungB(n, 2);
      const double &B4 = chungB(n, 3);
      const double &B5 = chungB(n, 4);
      const double &B6 = chungB(n, 5);
      const double &B7 = chungB(n, 6);

      double H2 =
          (B1 * (1.0 - exp(-B4 * Yy)) / Yy + B2 * G1 * exp(B5 * Yy) + B3 * G1) /
          (B1 * B4 + B2 + B3);

      double lambdak = lambda0 * (1.0 / H2 + B6 * Yy);
      double lambdap =
          (3.039e-4 * sqrt(Tcrit(n) / MW(n)) / pow(Vc, 2.0 / 3.0)) * B7 *
          pow(Yy, 2.0) * H2 * sqrt(Tr);

      // Compute final thermal conductivity, convert to SI units
      kappa_sp[n] = (lambdak + lambdap) * 418.68;
    }

    // Now every species' property is computed, generate mixture
    // values

    // viscosity mixture
    double mu = 0.0;
    for (int n = 0; n <= ns - 1; n++) {
      double phitemp = 0.0;
      for (int n2 = 0; n2 <= ns - 1; n2++) {
        double phi =
            pow((1.0 + sqrt(mu_sp[n] / mu_sp[n2] * sqrt(MW(n2) / MW(n)))),
                2.0) /
            (sqrt(8.0) * sqrt(1 + MW(n) / MW(n2)));
        phitemp += phi * X[n2];
      }
      mu += mu_sp[n] * X[n] / phitemp;
    }

    // thermal conductivity mixture
    double kappa = 0.0;
    {
      double sum1 = 0.0;
      double sum2 = 0.0;
      for (int n = 0; n <= ns - 1; n++) {
        sum1 += X[n] * kappa_sp[n];
        sum2 += X[n] / kappa_sp[n];
      }
      kappa = 0.5 * (sum1 + 1.0 / sum2);
    }

    // Set values of new properties
    // viscocity
    qt(0) = mu;
    // thermal conductivity
    qt(1) = kappa;
    // NOTE: Unity Lewis number approximation!
    for (int n = 0; n <= ns - 1; n++) {
      qt(2 + n) = kappa / (Q(0) * qh(1) * lewis(n));
    }
  }
};

PG_ABI void pgChungDenseGasUnityLewis(const chungDenseGasUnityLewis &k,
                                      const pgTiling &t) {
  forCells("Chung trans props unity Lewis", t, k);
}
