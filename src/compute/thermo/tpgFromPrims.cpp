#include "kernel.hpp"
#include "species.hpp"

PG_RANGE(cellCenters, halo = ng)
struct tpgFromPrims {
  cellCenterOut Q, qh;
  cellCenterInOut q;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // Updates all conserved quantities from primatives
    // Along the way, we need to compute mixture properties
    // gamma, cp, h, e, hi
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
    double hi[ns];
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

    // Compute Rmix
    Rmix = 0.0;
    for (int n = 0; n <= ns - 1; n++) {
      Rmix += Y[n] / MW(n);
    }
    Rmix *= Ru;

    // Update mixture properties
    h = 0.0;
    cp = 0.0;
    {
      const double u = log(T);
      for (int n = 0; n <= ns - 1; n++) {
        // cp/R and h/(RT), Horner in u
        double cpR = 0.0, hRT = 0.0;
        for (int m = cpPolyDegree(n) - 1; m >= 0; m--)
          cpR = cpR * u + cpPoly(n, m);
        for (int m = hPolyDegree(n) - 1; m >= 0; m--)
          hRT = hRT * u + hPoly(n, m);
        hRT += hRef(n) / T;
        const double Rn = Ru / MW(n);
        hi[n] = hRT * T * Rn;
        cp += cpR * Rn * Y[n];
        h += hi[n] * Y[n];
      }
    }

    // Compute mixuture enthalpy
    gamma = cp / (cp - Rmix);

    // Mixture speed of sound
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
      qh(5 + n) = hi[n];
    }
  }
};

PG_ABI void pgTpgFromPrims(const tpgFromPrims &k, const pgTiling &t) {
  forCells("Compute all conserved quantities from primatives via tgp", t, k);
}
