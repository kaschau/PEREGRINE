#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkos_types.hpp"

void rk2s1(block_ b, const double dt) {
  //-------------------------------------------------------------------------------------------|
  // Apply RK2 stage 1
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "rk2 stage 1", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.Q(i, j, k, l) += b.dQ(i, j, k, l) * dt;
      });
}

//-------------------------------------------------------------------------------------------|
// Apply RK2 stage 2
//-------------------------------------------------------------------------------------------|
void rk2s2(block_ b, const double dt) {
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "rk2 stage 2", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.Q(i, j, k, l) = 0.5 * b.rhs0(i, j, k, l) +
                          0.5 * (b.Q(i, j, k, l) + dt * b.dQ(i, j, k, l));
      });
}
