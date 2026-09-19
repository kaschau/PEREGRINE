#include "array"
#include "dualTime.hpp"
#include "kernel.hpp"
#include "vector"

PG_RANGE(cellCenters, components = ne)
struct dQdt {
  cellVecIn Q, Qn, Qnm1;
  cellVecInOut dQ;
  caseIn dt;
  KOKKOS_INLINE_FUNCTION void operator()(const int l) const {
    // the fpdtype time derivative, a source in pseudo time
    dQ(l) -= (3.0 * Q(l) - 4.0 * Qn(l) + Qnm1(l)) / (2 * dt());
  }
};

PG_ABI void pgDQdt(const dQdt &k, const pgTiling &t) {
  forCellsAndComponents("dQdt", t, k);
}
