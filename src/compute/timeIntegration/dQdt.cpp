#include "array"
#include "dualTime.hpp"
#include "kernel.hpp"
#include "vector"

PG_RANGE(cellCenters, components = ne)
struct dQdt {
  cellCenterIn Q, Qn, Qnm1;
  cellCenterInOut dQ;
  dims d;
  double dt;
  KOKKOS_INLINE_FUNCTION void operator()(const int l) const {
    const int ni = d->ni, nj = d->nj, nk = d->nk;
    //-------------------------------------------------------------------------------------------|
    // Add to dQ with real time derivative source term
    //-------------------------------------------------------------------------------------------|

    dQ(l) -= (3.0 * Q(l) - 4.0 * Qn(l) + Qnm1(l)) / (2 * dt);
  }
};

PG_ABI void pgDQdt(const dQdt &k, const pgTiling &t) {
  forCellsAndComponents("dQdt", t, k);
}
