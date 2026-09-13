#include "array"
#include "dualTime.hpp"
#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include "vector"
#include <Kokkos_Core.hpp>

PG_ABI void pgDTrk3s1(int count, pgOut *Q0_, pgIn *dQ_, pgIn *dtau_, pgOut *q_,
                      const pgDims *d) {
  for (int e = 0; e < count; e++) {
    auto Q0 = as4(Q0_[e]);
    auto dQ = as4(dQ_[e]);
    auto dtau = as3(dtau_[e]);
    auto q = as4(q_[e]);
    const int ni = d[e].ni, nj = d[e].nj, nk = d[e].nk;
    //-------------------------------------------------------------------------------------------|
    // Apply RK3 stage 1
    //-------------------------------------------------------------------------------------------|
    MDRange4 range_cc({ng, ng, ng, 0},
                      {ni + ng - 1, nj + ng - 1, nk + ng - 1, ne});
    Kokkos::parallel_for(
        "DTrk3 stage 1", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
          // store zeroth stage
          Q0(i, j, k, l) = q(i, j, k, l);
          q(i, j, k, l) += dtau(i, j, k) * dQ(i, j, k, l);
        });
  }
}
