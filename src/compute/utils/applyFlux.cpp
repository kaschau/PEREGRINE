#include "kernel.hpp"

// dQ is the flux difference across the cell along every axis, from the
// accumulated fluxes:
// begun here, or appended to what the chemistry source began it with
// (PG_FLUXES_APPEND, which the simulator bakes when a case has chemistry)
PG_RANGE(cellCenters, components = ne)
struct applyFlux {
  cellScalIn Jinv;
  iFaceStradVecIn iF;
  jFaceStradVecIn jF;
  kFaceStradVecIn kF;
  // read when appending
  cellVecInOut dQ;
  KOKKOS_INLINE_FUNCTION void operator()(const int l) const {
    const fpdtype f =
        (iF.L(l) + jF.L(l) + kF.L(l) - iF.R(l) - jF.R(l) - kF.R(l)) * Jinv;
#ifdef PG_FLUXES_APPEND
    dQ(l) += f;
#else
    dQ(l) = f;
#endif
  }
};

PG_ABI void pgApplyFlux(const applyFlux &k, const pgTiling &t) {
  forCellsAndComponents("Apply current fluxes to RHS", t, k);
}
