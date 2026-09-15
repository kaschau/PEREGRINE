#include "kernel.hpp"
#include "species.hpp"

PG_RANGE(cellCenters, halo = ng)
struct constantProps {
  cellCenterIn Q, q, qh;
  cellCenterOut qt;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    double Y[ns];
    double X[ns];

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
      for (int n = 0; n <= ns - 1; n++) {
        X[n] = Y[n] / MW(n) / mass;
      }
    }

    // viscosity mixture
    double mu = 0.0;
    for (int n = 0; n <= ns - 1; n++) {
      double phitemp = 0.0;
      for (int n2 = 0; n2 <= ns - 1; n2++) {
        double phi =
            pow((1.0 + sqrt(mu0(n) / mu0(n2) * sqrt(MW(n2) / MW(n)))), 2.0) /
            (sqrt(8.0) * sqrt(1 + MW(n) / MW(n2)));
        phitemp += phi * X[n2];
      }
      mu += mu0(n) * X[n] / phitemp;
    }

    // thermal conductivity mixture
    double kappa;
    {
      double sum1 = 0.0;
      double sum2 = 0.0;
      for (int n = 0; n <= ns - 1; n++) {
        sum1 += X[n] * kappa0(n);
        sum2 += X[n] / kappa0(n);
      }
      kappa = 0.5 * (sum1 + 1.0 / sum2);
    }

    // Set values of new properties
    // viscocity
    qt(0) = mu;
    // thermal conductivity
    qt(1) = kappa;
    // Diffusion coefficients mass
    // NOTE: Unity Lewis number approximation!
    for (int n = 0; n <= ns - 1; n++) {
      qt(2 + n) = kappa / (Q(0) * qh(1) * lewis(n));
    }
  }
};

PG_ABI void pgConstantProps(const constantProps &k, const pgTiling &t) {
  forCells("Const Props Transport", t, k);
}
