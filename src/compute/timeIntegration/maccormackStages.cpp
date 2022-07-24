#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkos_types.hpp"

void corrector(block_ b, const double dt) {
  //-------------------------------------------------------------------------------------------|
  // Apply Maccormack corrector stage
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "maccormack corrector", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.Q(i, j, k, l) =
            0.5 * (b.Q0(i, j, k, l) + b.Q(i, j, k, l) + dt * b.dQ(i, j, k, l));
      });
}
