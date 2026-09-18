#include "chemistry/rates.hpp"
#include "kernel.hpp"

// The net production rate of every species from the cell's state, kg /
// m^3 / s: the source term alone, for whatever integrates it.
PG_RANGE(cellCenters)
struct productionRates {
  cellCenterIn Q, q;
  cellCenterOut omega;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const fpdtype rho = Q(0);
    fpdtype Y[ns];
    chemistry::fractionsOf(Q, 1.0 / rho, Y);
    const auto s =
        chemistry::stateOf(rho, [&](const int n) { return Y[n]; }, q(1));
    fpdtype rate[ns];
    chemistry::netProduction(s, rate);
    for (int n = 0; n < ns; n++)
      omega(n) = rate[n];
  }
};

PG_ABI void pgProductionRates(const productionRates &k, const pgTiling &t) {
  forCells("net production rates", t, k);
}
