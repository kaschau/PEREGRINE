#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <math.h>

// The reductions, each over every block of the rank; python combines the
// ranks.

// 0 if any conserved quantity in the interior is not finite

PG_ABI int pgAllFinite(int count, const pgView *Q_, const pgDims *d) {
  int all = 1;
  for (int e = 0; e < count; e++) {
    auto Q = as4(Q_[e]);
    const int ni = d[e].ni, nj = d[e].nj, nk = d[e].nk;
    int allFinite;
    MDRange4 range_cc({ng, ng, ng, 0},
                      {ni + ng - 1, nj + ng - 1, nk + ng - 1, ne});
    Kokkos::parallel_reduce(
        "check nan", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int l,
                      int &finite) {
          finite = fmin(isfinite(Q(i, j, k, l)), finite);
        },
        Kokkos::Min<int>(allFinite));
    all = fmin(all, allFinite);
  }
  return all;
}

// the rank's max acoustic, convective and combined CFL speeds (speed/dx),
