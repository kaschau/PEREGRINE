#include "abi.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

PG_ABI void pgDQzero(const pgView *dQ_, const pgDims *d) {
  auto dQ = as4(*dQ_);

  //-------------------------------------------------------------------------------------------|
  // Zero out dQ
  //-------------------------------------------------------------------------------------------|
  Kokkos::deep_copy(dQ, 0.0);
}
