#include "kernel.hpp"

PG_RANGE(cellCenters, halo = ng)
struct cpgFromPrims {
  cellCenterOut Q, qh;
  cellCenterInOut q;
  record MW, cp0;
  double Ru;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // Updates all conserved quantities from primatives
    // Along the way, we need to compute mixture properties
    // gamma, cp, h, e
    // So we store these as well.

    const double &p = q(0);
    const double &u = q(1);
    const double &v = q(2);
    const double &w = q(3);
    const double &T = q(4);
    double Y[ns];

    double rho;
    double rhou, rhov, rhow;
    double e, tke, rhoE;
    double gamma, cp, h, c;
    double Rmix;

    // Compute nth species Y
    Y[ns - 1] = 1.0;
    double testSum = 0.0;
    for (int n = 0; n < ns - 1; n++) {
      q(5 + n) = fmax(fmin(q(5 + n), 1.0), 0.0);
      Y[n] = q(5 + n);
      Y[ns - 1] -= Y[n];
      testSum += Y[n];
    }

    // Renormalize if necessary
    if (testSum > 1.0) {
      Y[ns - 1] = 0.0;
      for (int n = 0; n < ns - 1; n++) {
        Y[n] /= testSum;
      }
    }

    // Update mixture properties
    Rmix = 0.0;
    cp = 0.0;
    for (int n = 0; n <= ns - 1; n++) {
      Rmix += Y[n] / MW(n);
      cp += Y[n] * cp0(n);
    }
    Rmix *= Ru;

    // Compute mixuture enthalpy
    h = cp * T;
    gamma = cp / (cp - Rmix);

    // Mixture speed of soung
    c = sqrt(gamma * Rmix * T);

    // Compute density
    rho = p / (Rmix * T);

    // Compute momentum
    rhou = rho * u;
    rhov = rho * v;
    rhow = rho * w;
    // Compuute TKE
    tke = 0.5 * (pow(u, 2.0) + pow(v, 2.0) + pow(w, 2.0)) * rho;

    // Compute internal, total, energy
    e = h - p / rho;
    rhoE = rho * e + tke;

    // Set values of new properties
    // Density
    Q(0) = rho;
    // Momentum
    Q(1) = rhou;
    Q(2) = rhov;
    Q(3) = rhow;
    // Total Energy
    Q(4) = rhoE;
    // Species mass
    for (int n = 0; n < ns - 1; n++) {
      Q(5 + n) = Y[n] * rho;
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

PG_ABI void pgCpgFromPrims(const cpgFromPrims &k, const pgTiling &t) {
  forCells("Compute all conserved quantities from primatives via cpg", t, k);
}
