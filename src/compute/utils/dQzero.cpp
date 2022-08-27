#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkos_types.hpp"

void dQzero(block_ &b) {

  //-------------------------------------------------------------------------------------------|
  // Zero out dQ
  //-------------------------------------------------------------------------------------------|
  Kokkos::deep_copy(b.dQ, 0.0);
}
