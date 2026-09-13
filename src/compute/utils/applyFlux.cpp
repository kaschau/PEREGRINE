#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include <Kokkos_Core.hpp>

PG_ABI void pgApplyFlux(int count, pgIn *Jinv_, pgOut *dQ_, pgIn *iF_,
                        pgIn *jF_, pgIn *kF_, const pgDims *d) {
  for (int e = 0; e < count; e++) {
    auto Jinv = as3(Jinv_[e]);
    auto dQ = as4(dQ_[e]);
    auto iF = as4(iF_[e]);
    auto jF = as4(jF_[e]);
    auto kF = as4(kF_[e]);
    const int ni = d[e].ni, nj = d[e].nj, nk = d[e].nk;

    //-------------------------------------------------------------------------------------------|
    // Apply fluxes to cc range
    //-------------------------------------------------------------------------------------------|
    MDRange4 range_cc({ng, ng, ng, 0},
                      {ni + ng - 1, nj + ng - 1, nk + ng - 1, ne});
    Kokkos::parallel_for(
        "Apply current fluxes to RHS", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
          // Add fluxes to RHS
          dQ(i, j, k, l) += (iF(i, j, k, l) + jF(i, j, k, l) + kF(i, j, k, l)) *
                            Jinv(i, j, k);

          dQ(i, j, k, l) -=
              (iF(i + 1, j, k, l) + jF(i, j + 1, k, l) + kF(i, j, k + 1, l)) *
              Jinv(i, j, k);
        });
  }
}
