#include "array"
#include "dualTime.hpp"
#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include "vector"
#include <Kokkos_Core.hpp>

PG_ABI void pgDQdt(int count, pgIn *Q_, pgIn *Qn_, pgIn *Qnm1_, pgOut *dQ_,
                   const pgDims *d, double dt) {
  for (int e = 0; e < count; e++) {
    auto Q = as4(Q_[e]);
    auto Qn = as4(Qn_[e]);
    auto Qnm1 = as4(Qnm1_[e]);
    auto dQ = as4(dQ_[e]);
    const int ni = d[e].ni, nj = d[e].nj, nk = d[e].nk;
    //-------------------------------------------------------------------------------------------|
    // Add to dQ with real time derivative source term
    //-------------------------------------------------------------------------------------------|
    MDRange4 range_cc({ng, ng, ng, 0},
                      {ni + ng - 1, nj + ng - 1, nk + ng - 1, ne});
    Kokkos::parallel_for(
        "dQdt", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
          dQ(i, j, k, l) -=
              (3.0 * Q(i, j, k, l) - 4.0 * Qn(i, j, k, l) + Qnm1(i, j, k, l)) /
              (2 * dt);
        });
  }
}
