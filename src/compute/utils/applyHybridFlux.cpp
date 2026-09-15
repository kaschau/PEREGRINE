#include "kernel.hpp"

PG_RANGE(cellCenters, components = ne)
struct applyHybridFlux {
  cellCenterIn Jinv, iF, jF, kF, phi;
  cellCenterInOut dQ;
  dims d;
  double primary;
  KOKKOS_INLINE_FUNCTION void operator()(const int l) const {
    const int ni = d->ni, nj = d->nj, nk = d->nk;
    //-------------------------------------------------------------------------------------------|
    //-------------------------------------------------------------------------------------------|

    // Compute switch on face
    double iFphi = fmax(phi(0), phi(-I, 0));
    double iFphi1 = fmax(phi(0), phi(+I, 0));
    double jFphi = fmax(phi(1), phi(-J, 1));
    double jFphi1 = fmax(phi(1), phi(+J, 1));
    double kFphi = fmax(phi(2), phi(-K, 2));
    double kFphi1 = fmax(phi(2), phi(+K, 2));

    double dPrimary = 2.0 * primary - 1.0;

    // Add fluxes to RHS
    // format is F_primary*(1-switch) + F_secondary*(switch)
    // so when switch == 0, we dont switch from primary
    //    when switch == 1 we completely switch
    dQ(l) += (iF(l) * (primary - iFphi * dPrimary) +
              jF(l) * (primary - jFphi * dPrimary) +
              kF(l) * (primary - kFphi * dPrimary)) *
             Jinv();

    dQ(l) -= (iF(+I, l) * (primary - iFphi1 * dPrimary) +
              jF(+J, l) * (primary - jFphi1 * dPrimary) +
              kF(+K, l) * (primary - kFphi1 * dPrimary)) *
             Jinv();
  }
};

PG_ABI void pgApplyHybridFlux(const applyHybridFlux &k, const pgTiling &t) {
  forCellsAndComponents("Apply hybrid fluxes to RHS", t, k);
}
