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
  cellVecIn Q, q, qh;
  cellVecOut qt;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // the mole fractions off the conserved state
    fpdtype X[ns];
    const fpdtype rhoinv = 1.0 / Q(0);
    const fpdtype MWmix = moleFractions(massFractions(Q, rhoinv), X);
    const fpdtype T = q(1);

    {
      fpdtype sqrtMu[ns];
      for (int n = 0; n <= ns - 1; n++) {
        sqrtMu[n] = sqrt(mu0(n));
      }
      qt(0) = mixingRule::viscosity(X, sqrtMu);
    }
    fpdtype kappaSp[ns];
    for (int n = 0; n <= ns - 1; n++) {
      kappaSp[n] = kappa0(n);
    }
    const fpdtype kappa = mixtureConductivity(X, kappaSp);
    qt(1) = kappa;

    fpdtype D[ns];
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
