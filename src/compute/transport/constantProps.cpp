#include "conserved.hpp"
#include "diffusion.hpp"
#include "kernel.hpp"
#include "mixing.hpp"
#include "mixingRule.hpp"

// Constant species properties: each species' viscosity and conductivity as
// the mixture gives them, mixed by Wilke's rule and the series-parallel
// mean; the species diffusion coefficients from the case's diffusion model.
PG_RANGE(cellCenters)
struct constantProps {
  cellCenterIn Q, q, qh;
  cellCenterOut qt;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // the mole fractions off the conserved state
    double X[ns];
    const double rhoinv = 1.0 / Q(0);
    const double MWmix = moleFractions(massFractions(Q, rhoinv), X);
    const double T = q(1);

    {
      double sqrtMu[ns];
      for (int n = 0; n <= ns - 1; n++) {
        sqrtMu[n] = sqrt(mu0(n));
      }
      qt(0) = mixingRule::viscosity(X, sqrtMu);
    }
    double kappaSp[ns];
    for (int n = 0; n <= ns - 1; n++) {
      kappaSp[n] = kappa0(n);
    }
    const double kappa = mixtureConductivity(X, kappaSp);
    qt(1) = kappa;

    double D[ns];
    diffusion::coefficients({q(0), T, log(T), rhoinv, qh(1), kappa, MWmix, X},
                            D);
    for (int n = 0; n <= ns - 1; n++) {
      qt(2 + n) = D[n];
    }
  }
};

PG_ABI void pgConstantProps(const constantProps &k, const pgTiling &t) {
  forCells("Const Props Transport", t, k);
}
