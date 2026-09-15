#include "kernel.hpp"
#include "species.hpp"

PG_RANGE(cellCenters, halo = ng)
struct kineticTheory {
  cellCenterIn q;
  cellCenterOut qt;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // poly'l degree

    const double &p = q(0);
    const double &T = q(4);
    double Y[ns];
    double X[ns];
    double mu_sp[ns] = {};
    double kappa_sp[ns] = {};
    double Dij[ns][ns] = {};
    double D[ns] = {};

    // check for pure fluid, this int will represent
    // the index of the species array in which we
    // have a pure fluid.
    int pure = -1;

    // Compute nth species Y
    Y[ns - 1] = 1.0;
    for (int n = 0; n < ns - 1; n++) {
      Y[n] = q(5 + n);
      Y[ns - 1] -= Y[n];
    }

    // Update mixture properties
    // Mole fractions
    double MWmix = 0.0;
    {
      double mass = 0.0;
      for (int n = 0; n <= ns - 1; n++) {
        mass += Y[n] / MW(n);
      }

      // Mean molecular weight, mole fraction
      for (int n = 0; n <= ns - 1; n++) {
        X[n] = Y[n] / MW(n) / mass;
        MWmix += X[n] * MW(n);
        if (X[n] == 1.0) {
          pure = n;
          break;
        }
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
      for (int n2 = n; n2 <= ns - 1; n2++) {
        Dij[n][n2] = 0.0;
        for (int m = dijDegree(n, n2) - 1; m >= 0; m--)
          Dij[n][n2] = Dij[n][n2] * u + dij(n, n2, m);
      }

      // Set to the correct dimensions
      mu_sp[n] *= sqrtsqrt_T;
      mu_sp[n] *= mu_sp[n];
      kappa_sp[n] *= sqrt_T;
      const double T_3o2 = T * sqrt_T;
      for (int n2 = n; n2 <= ns - 1; n2++) {
        Dij[n][n2] *= T_3o2;
        Dij[n2][n] = Dij[n][n2];
      }
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
            (sqrt(8.0) * sqrt(1.0 + MW(n) / MW(n2)));
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

    // mass diffusion coefficient mixture
    if (pure == -1) {
      for (int n = 0; n <= ns - 1; n++) {
        double sum1 = 0.0;
        double sum2 = 0.0;
        for (int n2 = 0; n2 <= ns - 1; n2++) {
          if (n == n2) {
            continue;
          }
          sum1 += X[n2] / Dij[n][n2];
          sum2 += X[n2] * MW(n2) / Dij[n][n2];
        }
        // Account for pressure
        sum1 *= p;
        sum2 *= p * X[n] / (MWmix - MW(n) * X[n]);
        D[n] = 1.0 / (sum1 + sum2);
      }
    }

    // Set values of new properties
    // viscocity
    qt(0) = mu;
    // thermal conductivity
    qt(1) = kappa;
    // Diffusion coefficients mass
    for (int n = 0; n <= ns - 1; n++) {
      qt(2 + n) = D[n];
    }
  }
};

PG_ABI void pgKineticTheory(const kineticTheory &k, const pgTiling &t) {
  forCells("Kinetic Theory trans props", t, k);
}
