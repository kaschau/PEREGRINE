#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkos_types.hpp"
#include "vector"
#include <math.h>

int checkNan(std::vector<block_> mb) {

  //-------------------------------------------------------------------------------------------|
  // Check for nans and infs in the solution array.
  //-------------------------------------------------------------------------------------------|
  int foundNan;
  int nanDetected = 0;

  for (const block_ b : mb) {
    MDRange4 range_cc({b.ng, b.ng, b.ng, 0}, {b.ni + b.ng - 1, b.nj + b.ng - 1,
                                              b.nk + b.ng - 1, b.ne});
    Kokkos::parallel_reduce(
        "check nan", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int l, int &isNan) {
          isNan = fmax(isnan(b.Q(i, j, k, l)), isNan);
        },
        Kokkos::Max<int>(foundNan));
  }

  nanDetected = std::max(nanDetected, foundNan);

  return nanDetected;
}
