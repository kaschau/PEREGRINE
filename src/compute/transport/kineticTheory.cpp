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
  cellVecIn Q, q, qh;
  cellVecOut qt;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const fpdtype p = q(0);
    const fpdtype T = q(1);

    // the mole fractions off the conserved state
    fpdtype X[ns];
    const fpdtype rhoinv = 1.0 / Q(0);
    const fpdtype MWmix = moleFractions(massFractions(Q, rhoinv), X);

    // each species' fit, Horner in u = ln T: the viscosity's is of
    // sqrt(mu) / T^(1/4), the conductivity's of kappa / sqrt(T)
    const fpdtype u = log(T);
    const fpdtype sqrtT = sqrt(T);
    const fpdtype sqrtsqrtT = sqrt(sqrtT);
    // each species' array lives only until its mixture rule has run
    {
      fpdtype sqrtMu[ns];
      for (int n = 0; n <= ns - 1; n++) {
        fpdtype mu = 0.0;
        for (int m = muPolyTerms - 1; m >= 0; m--)
          mu = mu * u + muPoly(n, m);
        sqrtMu[n] = mu * sqrtsqrtT;
      }
      qt(0) = mixingRule::viscosity(X, sqrtMu);
    }
    fpdtype kappaSp[ns];
    for (int n = 0; n <= ns - 1; n++) {
      fpdtype kappa = 0.0;
      for (int m = kappaPolyTerms - 1; m >= 0; m--)
        kappa = kappa * u + kappaPoly(n, m);
      kappaSp[n] = kappa * sqrtT;
    }
    const fpdtype kappa = mixtureConductivity(X, kappaSp);
    qt(1) = kappa;

    fpdtype D[ns];
    diffusion::coefficients({p, T, u, rhoinv, qh(1), kappa, MWmix, X}, D);
    for (int n = 0; n <= ns - 1; n++) {
      qt(2 + n) = D[n];
    }
  }
};

PG_ABI void pgKineticTheory(const kineticTheory &k, const pgTiling &t) {
  forCells("Kinetic Theory trans props", t, k);
}
