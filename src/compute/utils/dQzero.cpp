#include "abi.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

PG_ABI void pgDQzero(int count, const pgView *dQ_, const pgDims *d) {
  for (int e = 0; e < count; e++) {
    auto dQ = as4(dQ_[e]);

    //-------------------------------------------------------------------------------------------|
    // Zero out dQ
    //-------------------------------------------------------------------------------------------|
    Kokkos::deep_copy(dQ, 0.0);
  }
}
