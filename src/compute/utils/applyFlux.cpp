#include "kernel.hpp"

// dQ is the flux difference of every face, from the accumulated fluxes
PG_RANGE(interior, ne)
struct applyFlux {
  in Jinv, iF, jF, kF;
  out dQ;
  KOKKOS_INLINE_FUNCTION void operator()(const int l) const {
    dQ(l) =
        (iF(l) + jF(l) + kF(l) - iF(+I, l) - jF(+J, l) - kF(+K, l)) * Jinv();
  }
};

PG_ABI void pgApplyFlux(const applyFlux &k, const pgTiling &t) {
  forCellsAndComponents("Apply current fluxes to RHS", t, k);
}
