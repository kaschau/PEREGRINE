#include "kernel.hpp"

// the production rates as the species equations' source
PG_RANGE(cellCenters)
struct applyOmega {
  cellCenterIn omega;
  cellCenterInOut dQ;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    for (int n = 0; n < ns - 1; n++)
      dQ(5 + n) += omega(n);
  }
};

PG_ABI void pgApplyOmega(const applyOmega &k, const pgTiling &t) {
  forCells("apply production rates", t, k);
}
