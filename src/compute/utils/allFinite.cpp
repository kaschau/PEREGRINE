#include "kernel.hpp"

// 0 if any conserved quantity in the interior of any block is not finite;
// python combines the ranks.
PG_RANGE(interior, ne)
struct allFinite {
  in Q;
  KOKKOS_INLINE_FUNCTION void operator()(const int l, int &finite) const {
    finite = fmin(isfinite(Q(l)), finite);
  }
};

PG_ABI int pgAllFinite(const allFinite &k, const pgTiling &t) {
  int all = 1;
  reduceCellsAndComponents("check nan", t, k, Kokkos::Min<int>(all));
  return all;
}
