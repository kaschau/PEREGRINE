#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <math.h>

// The reductions, each over every block of the rank; python combines the
// ranks.

// 0 if any conserved quantity in the interior is not finite

// and Q0 (which holds primitives under dual time), into rMax[ne], rSum[ne]
PG_ABI void pgResidual(int count, pgIn *q_, pgIn *Q0_, const pgDims *d,
                       double *rMax, double *rSum) {
  for (int m = 0; m < ne; m++)
    rMax[m] = rSum[m] = 0.0;
  for (int e = 0; e < count; e++) {
    auto q = as4(q_[e]);
    auto Q0 = as4(Q0_[e]);
    const int ni = d[e].ni, nj = d[e].nj, nk = d[e].nk;
    MDRange3 range_cc({ng, ng, ng}, {ni + ng - 1, nj + ng - 1, nk + ng - 1});
    for (int m = 0; m < ne; m++) {
      double rmax, rsum;
      Kokkos::parallel_reduce(
          "residual", range_cc,
          KOKKOS_LAMBDA(const int i, const int j, const int k, double &resMax,
                        double &resSum) {
            const double res = abs(q(i, j, k, m) - Q0(i, j, k, m));
            resMax = fmax(res, resMax);
            resSum += res * res;
          },
          Kokkos::Max<double>(rmax), Kokkos::Sum<double>(rsum));
      rMax[m] = fmax(rMax[m], rmax);
      rSum[m] += rsum;
    }
  }
}
