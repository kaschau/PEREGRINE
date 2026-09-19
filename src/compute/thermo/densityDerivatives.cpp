#include "eos.hpp"
#include "kernel.hpp"

// The density's derivatives at every cell, as the eos gives them: by p, by
// T, and by each of the first ns - 1 mass fractions with the last taking
// up the change -- what the dual time preconditioning linearizes with,
// written out so they can be checked against the state itself.
PG_RANGE(cellCenters)
struct densityDerivatives {
  cellVecIn Q, q;
  cellVecOut qj;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const fpdtype rho = Q(0);
    fpdtype Y[ns], rho_Y[ns];
    massFractions(Q, 1.0 / rho, Y);
    fpdtype rho_p, rho_T;
    eos::densityDerivatives(
        q(0), q(1), rho, [&](const int n) { return Y[n]; }, rho_p, rho_T,
        rho_Y);
    qj(0) = rho_p;
    qj(1) = rho_T;
    for (int n = 0; n < ns - 1; n++)
      qj(2 + n) = rho_Y[n];
  }
};

PG_ABI void pgDensityDerivatives(const densityDerivatives &k,
                                 const pgTiling &t) {
  forCells("density derivatives", t, k);
}
