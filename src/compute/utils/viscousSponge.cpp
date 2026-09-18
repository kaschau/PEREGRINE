#include "kernel.hpp"

// A sponge: the viscosity raised toward :mult: times itself along a line,
// from one at :start: (the origin's projection on the sponge's unit normal)
// to mult a :length: further, and held there.
PG_RANGE(cellCenters)
struct viscousSponge {
  cellCenterIn cells;
  cellCenterInOut qt;
  double nx, ny, nz, start, length, mult;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const double along =
        (cells(0) * nx + cells(1) * ny + cells(2) * nz - start) / length;
    qt(0) *= 1.0 + fmin(fmax(0.0, along * (mult - 1.0)), mult - 1.0);
  }
};

PG_ABI void pgViscousSponge(const viscousSponge &k, const pgTiling &t) {
  forCells("Apply viscous sponge", t, k);
}
