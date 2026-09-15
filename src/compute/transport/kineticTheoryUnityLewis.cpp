#include "kernel.hpp"
#include "species.hpp"

PG_RANGE(cellCenters, halo = ng)
struct kineticTheoryUnityLewis {
  cellCenterIn Q, q, qh;
  cellCenterOut qt;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // poly'l degree

    const double &T = q(4);
    double Y[ns];
    double X[ns];
    double mu_sp[ns] = {};
    double kappa_sp[ns] = {};

    // Compute nth species Y
    Y[ns - 1] = 1.0;
    for (int n = 0; n < ns - 1; n++) {
      Y[n] = q(5 + n);
      Y[ns - 1] -= Y[n];
    }

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

    // Evaluate all property polynomials, Horner in u = ln T
    const double u = log(T);
    const double sqrt_T = sqrt(T);
    const double sqrtsqrt_T = sqrt(sqrt_T);
    for (int n = 0; n <= ns - 1; n++) {
      // the scratch persists between cells; Horner starts from zero
      mu_sp[n] = 0.0;
      kappa_sp[n] = 0.0;
      for (int m = muPolyDegree(n) - 1; m >= 0; m--)
        mu_sp[n] = mu_sp[n] * u + muPoly(n, m);
      for (int m = kappaPolyDegree(n) - 1; m >= 0; m--)
        kappa_sp[n] = kappa_sp[n] * u + kappaPoly(n, m);

      // Set to the correct dimensions
      // the fit is of sqrt(mu)/T^(1/4), so undo both
      mu_sp[n] *= sqrtsqrt_T;
      mu_sp[n] *= mu_sp[n];
      kappa_sp[n] *= sqrt_T;
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

PG_ABI void pgKineticTheoryUnityLewis(const kineticTheoryUnityLewis &k,
                                      const pgTiling &t) {
  forCells("Kinetic theory unity lewis", t, k);
}
