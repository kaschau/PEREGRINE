#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

PG_ABI void pgRk4s4(int count, pgOut *Q_, pgIn *Q0_, pgIn *Q1_, pgIn *Q2_,
                    pgIn *Q3_, pgIn *dQ_, const pgDims *d, double dt) {
  for (int e = 0; e < count; e++) {
    auto Q = as4(Q_[e]);
    auto Q0 = as4(Q0_[e]);
    auto Q1 = as4(Q1_[e]);
    auto Q2 = as4(Q2_[e]);
    auto Q3 = as4(Q3_[e]);
    auto dQ = as4(dQ_[e]);
    const int ni = d[e].ni, nj = d[e].nj, nk = d[e].nk;
    //-------------------------------------------------------------------------------------------|
    // Apply RK4 stage 4
    //-------------------------------------------------------------------------------------------|
    MDRange4 range_cc({ng, ng, ng, 0},
                      {ni + ng - 1, nj + ng - 1, nk + ng - 1, ne});
    Kokkos::parallel_for(
        "rk4 stage 4", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
          Q(i, j, k, l) =
              Q0(i, j, k, l) + (Q1(i, j, k, l) + 2.0 * Q2(i, j, k, l) +
                                2.0 * Q3(i, j, k, l) + dt * dQ(i, j, k, l)) /
                                   6.0;
        });
  }
}
