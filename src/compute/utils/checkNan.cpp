#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkos_types.hpp"
#include "vector"
#include <math.h>

int checkNan(std::vector<block_> mb) {

  //-------------------------------------------------------------------------------------------|
  // Check for nans and infs in the solution array.
  //-------------------------------------------------------------------------------------------|
  int allFinite;
  int foundNan = 0; // <-- foundNan ends up as q when nans are detected

  for (const block_ b : mb) {
    MDRange4 range_cc({b.ng, b.ng, b.ng, 0}, {b.ni + b.ng - 1, b.nj + b.ng - 1,
                                              b.nk + b.ng - 1, b.ne});
    Kokkos::parallel_reduce(
        "check nan", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int l,
                      int &finite) {
          finite = fmin(isfinite(b.Q(i, j, k, l)), finite);
        },
        Kokkos::Min<int>(allFinite));

    if (allFinite == 0){
        foundNan = 1;
        break;
    }
  }

  return foundNan;
}
