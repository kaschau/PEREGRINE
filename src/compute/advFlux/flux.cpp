#include "faces.hpp"

// One advective flux composed by the jit: the reconstruction, PG_RECONSTRUCT
// (piecewiseConstant, muscl), is the kernel struct, forced in after the Riemann
// solver it hands its two states to, PG_RIEMANN (rusanov, hllc, ausmPlusUp),
// and with muscl the limiter it slopes by, PG_LIMITER.
#ifndef PG_RECONSTRUCT
#error "a flux is composed: -DPG_RECONSTRUCT and -DPG_RIEMANN from the jit"
#endif
#define PG_STRING_(a) #a
#define PG_STRING(a) PG_STRING_(a)

using flux = PG_RECONSTRUCT;

PG_RANGE(cellFaces)
PG_ABI void pgFlux(const flux &k, const pgTiling &t) {
  forCells(PG_STRING(PG_RECONSTRUCT) " " PG_STRING(PG_RIEMANN) " fluxes", t, k);
}
