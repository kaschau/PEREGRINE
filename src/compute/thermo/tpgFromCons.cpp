#include "kernel.hpp"

PG_RANGE(cells)
struct tpgFromCons {
  inout Q, q;
  out qh;
  record MW, cpPoly, hPoly, hRef;
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
    double e, tke;
    double T;
    double Y[ns];
    double gamma, cp, h, c;
    double hi[ns];
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

    // Compute Rmix
    Rmix = 0.0;
    for (int n = 0; n <= ns - 1; n++) {
      Rmix += Y[n] / MW(n);
    }
    Rmix *= Ru;

    // Iterate on to find temperature
    int nitr = 0, maxitr = 100;
    double tol = 1e-8;
    double error = 1e100;
    // Newtons method to find T
    T = (q(4) < 1.0) ? 300.0 : q(4); // Initial guess of T
    while ((abs(error) > tol) && (nitr < maxitr)) {
      h = 0.0;
      cp = 0.0;
      {
        const double u = log(T);
        for (int n = 0; n <= ns - 1; n++) {
          double cpR = 0.0, hRT = 0.0;
          for (int m = cpPoly.extent(1) - 1; m >= 0; m--)
            cpR = cpR * u + cpPoly(n, m);
          for (int m = hPoly.extent(1) - 1; m >= 0; m--)
            hRT = hRT * u + hPoly(n, m);
          hRT += hRef(n) / T;
          const double Rn = Ru / MW(n);
          hi[n] = hRT * T * Rn;
          cp += cpR * Rn * Y[n];
          h += hi[n] * Y[n];
        }
      }

      error = e - (h - Rmix * T);
      T = T - error / (-cp + Rmix);
      nitr += 1;
    }

    // Compute mixuture pressure
    p = rho * Rmix * T;
    // Compute mixture gamma
    gamma = cp / (cp - Rmix);

    // Mixture speed of sound
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
      qh(5 + n) = hi[n];
    }
  }
};

PG_ABI void pgTpgFromCons(const tpgFromCons &k, const pgTiling &t) {
  forCells("Compute primatives from conserved quantities via tpg", t, k);
}
