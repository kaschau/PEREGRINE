#include "kernel.hpp"

PG_RANGE(cellCenters, halo = ng)
struct cpgFromCons {
  cellCenterInOut Q;
  cellCenterOut q, qh;
  record MW, cp0;
  double Ru;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // Updates all primatives from conserved quantities
    // Along the way, we need to compute mixture properties
    // gamma, cp, h, e, hi
    // So we store these as well.

    const double &rho = Q(0);
    const double &rhou = Q(1);
    const double &rhov = Q(2);
    const double &rhow = Q(3);
    const double &rhoE = Q(4);

    double p;
    double T;
    double e, tke;
    double Y[ns];
    double gamma, cp, h, c;
    double Rmix;

    // Compute TKE
    tke = 0.5 * (pow(rhou, 2.0) + pow(rhov, 2.0) + pow(rhow, 2.0)) / rho;

    // Compute species mass fraction
    Y[ns - 1] = 1.0;
    double testSum = 0.0;
    for (int n = 0; n < ns - 1; n++) {
      Q(5 + n) = fmax(fmin(Q(5 + n), Q(0)), 0.0);
      Y[n] = Q(5 + n) / Q(0);
      Y[ns - 1] -= Y[n];
      testSum += Y[n];
    }

    // Renormalize if necessary
    if (testSum > 1.0) {
      Y[ns - 1] = 0.0;
      for (int n = 0; n < ns - 1; n++) {
        Y[n] /= testSum;
        Q(5 + n) = Y[n] * Q(0);
      }
    }

    // Internal energy
    e = (rhoE - tke) / rho;

    // Compute mixuture cp
    Rmix = 0.0;
    cp = 0.0;
    for (int n = 0; n <= ns - 1; n++) {
      Rmix += Y[n] / MW(n);
      cp += Y[n] * cp0(n);
    }
    Rmix *= Ru;

    // Compute mixuture temperature,pressure
    T = e / (cp - Rmix);
    p = rho * Rmix * T;

    // Compute mixture enthalpy
    h = e + p / rho;
    gamma = cp / (cp - Rmix);

    // Mixture speed of soung
    c = sqrt(gamma * Rmix * T);

    // Set values of new properties
    // Pressure, temperature, Y
    q(0) = p;
    q(1) = rhou / rho;
    q(2) = rhov / rho;
    q(3) = rhow / rho;
    q(4) = T;
    for (int n = 0; n < ns - 1; n++) {
      q(5 + n) = Y[n];
    }
    // gamma,cp,h,c,e,hi
    qh(0) = gamma;
    qh(1) = cp;
    qh(2) = rho * h;
    qh(3) = c;
    qh(4) = rho * e;
    for (int n = 0; n <= ns - 1; n++) {
      qh(5 + n) = T * cp0(n);
    }
  }
};

PG_ABI void pgCpgFromCons(const cpgFromCons &k, const pgTiling &t) {
  forCells("Compute primatives from conserved quantities via cpg", t, k);
}
