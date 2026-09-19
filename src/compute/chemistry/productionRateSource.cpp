#include "chemistry/rates.hpp"
#include "kernel.hpp"

// dQ begun with the net production rates of the cell's state, as they are:
// the explicit chemistry source.
PG_RANGE(cellCenters)
struct productionRateSource {
  cellVecIn Q, q;
  cellVecOut dQ;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    fpdtype rate[ns];
    chemistry::productionRatesOf(Q, q, rate);
    chemistry::beginWithSource(dQ, rate);
  }
};

PG_ABI void pgProductionRateSource(const productionRateSource &k,
                                   const pgTiling &t) {
  forCells("production rate source", t, k);
}
