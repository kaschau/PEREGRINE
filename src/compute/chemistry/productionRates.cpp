#include "chemistry/rates.hpp"
#include "kernel.hpp"

// The net production rate of every species from the cell's state, kg /
// m^3 / s, kept: the source term alone, for whatever wants it as it is.
PG_RANGE(cellCenters)
struct productionRates {
  cellVecIn Q, q;
  cellVecOut omega;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    fpdtype rate[ns];
    chemistry::productionRatesOf(Q, q, rate);
    for (int n = 0; n < ns; n++)
      omega(n) = rate[n];
  }
};

PG_ABI void pgProductionRates(const productionRates &k, const pgTiling &t) {
  forCells("net production rates", t, k);
}
