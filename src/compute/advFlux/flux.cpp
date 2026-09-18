#include "advFlux/fluxOut.hpp"
#include "faces.hpp"

// One advective flux composed by the jit. The kernel struct is PG_BASE: the
// reconstruction, PG_RECONSTRUCT (piecewiseConstant, muscl), which holds the
// columns and makes the two states at the face, and with shock capturing
// the switch on it, which weighs the face. PG_PRIMARY (a Riemann solver:
// rusanov, hllc, ausmPlusUp; a central scheme: KEPaEC) makes the flux of
// the states; with a switch, PG_SECONDARY is blended in by the weight,
// (1 - w) primary + w secondary. Each is forced in ahead by the jit, and
// with muscl the limiter it slopes by, PG_LIMITER.
#ifndef PG_BASE
#error                                                                         \
    "a flux is composed: -DPG_BASE, -DPG_RECONSTRUCT and -DPG_PRIMARY from the jit"
#endif
#define PG_STRING_(a) #a
#define PG_STRING(a) PG_STRING_(a)

PG_RANGE(cellFaces)
struct flux : PG_BASE {
  using base = PG_BASE;
  KOKKOS_INLINE_FUNCTION void operator()() const {
#ifdef PG_SECONDARY
    const double w = this->weight();
    PG_PRIMARY::flux(*this, this->A, weighted{this->F, 1.0 - w});
    PG_SECONDARY::flux(*this, this->A, added{this->F, w});
#else
    PG_PRIMARY::flux(*this, this->A, this->F);
#endif
  }
};

PG_ABI void pgFlux(const flux &k, const pgTiling &t) {
  forCells(PG_STRING(PG_BASE) " " PG_STRING(PG_PRIMARY) " fluxes", t, k);
}
