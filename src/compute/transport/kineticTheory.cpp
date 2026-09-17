#include "conserved.hpp"
#include "diffusion.hpp"
#include "kernel.hpp"
#include "mixing.hpp"
#include "mixingRule.hpp"

// Kinetic theory transport: each species' viscosity and conductivity from
// its fit in ln T, mixed by Wilke's rule and the series-parallel mean; the
// species diffusion coefficients from the case's diffusion model.
PG_RANGE(cellCenters)
struct kineticTheory {
  cellCenterIn Q, q, qh;
  cellCenterOut qt;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const double p = q(0);
    const double T = q(1);

    // the mole fractions off the conserved state
    double X[ns];
    const double rhoinv = 1.0 / Q(0);
    const double MWmix = moleFractions(massFractions(Q, rhoinv), X);

    // each species' fit, Horner in u = ln T: the viscosity's is of
    // sqrt(mu) / T^(1/4), the conductivity's of kappa / sqrt(T)
    const double u = log(T);
    const double sqrtT = sqrt(T);
    const double sqrtsqrtT = sqrt(sqrtT);
    // each species' array lives only until its mixture rule has run
    {
      double sqrtMu[ns];
      for (int n = 0; n <= ns - 1; n++) {
        double mu = 0.0;
        for (int m = muPolyTerms - 1; m >= 0; m--)
          mu = mu * u + muPoly(n, m);
        sqrtMu[n] = mu * sqrtsqrtT;
      }
      qt(0) = mixingRule::viscosity(X, sqrtMu);
    }
    double kappaSp[ns];
    for (int n = 0; n <= ns - 1; n++) {
      double kappa = 0.0;
      for (int m = kappaPolyTerms - 1; m >= 0; m--)
        kappa = kappa * u + kappaPoly(n, m);
      kappaSp[n] = kappa * sqrtT;
    }
    const double kappa = mixtureConductivity(X, kappaSp);
    qt(1) = kappa;

    double D[ns];
    diffusion::coefficients({p, T, u, rhoinv, qh(1), kappa, MWmix, X}, D);
    for (int n = 0; n <= ns - 1; n++) {
      qt(2 + n) = D[n];
    }
  }
};

PG_ABI void pgKineticTheory(const kineticTheory &k, const pgTiling &t) {
  forCells("Kinetic Theory trans props", t, k);
}
