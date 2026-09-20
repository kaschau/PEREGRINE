#include "chemistry/jacobian.hpp"
#include "kernel.hpp"

// The entries of the species source's Jacobian the case's rung evaluates,
// a carried species' row over the primitives' columns, kept as a matrix
// with the rest of it zero: what the pseudo system of dual time is
// preconditioned with, written out so they can be checked against the
// rates themselves.
#ifndef PG_CHEMISTRY_JACOBIAN
#error "the source's Jacobian takes chemistryJacobian from the config"
#endif

PG_RANGE(cellCenters)
struct productionRateJacobian {
  cellVecIn Q, q;
  cellMatOut omegaJ;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    for (int j = 0; j < ns - 1; j++)
      for (int col = 0; col < ne; col++)
        omegaJ(j, col) = 0.0;
    chemistry::jacobianOf<chemistry::PG_CHEMISTRY_JACOBIAN>(
        Q, q, [&](const int j, const int col, const fpdtype v) {
          omegaJ(j, col) += v;
        });
  }
};

PG_ABI void pgProductionRateJacobian(const productionRateJacobian &k,
                                     const pgTiling &t) {
  forCells("production rate Jacobian", t, k);
}
