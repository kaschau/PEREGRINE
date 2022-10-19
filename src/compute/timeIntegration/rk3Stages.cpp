#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkosTypes.hpp"

void rk3s1(block_ &b, const double &dt) {
  //-------------------------------------------------------------------------------------------|
  // Apply RK3 stage 1
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "rk3 stage 1", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.Q(i, j, k, l) = b.Q0(i, j, k, l) + dt * b.dQ(i, j, k, l);
      });
}

void rk3s2(block_ &b, const double &dt) {
  //-------------------------------------------------------------------------------------------|
  // Apply RK3 stage 2
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "rk3 stage 2", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.Q(i, j, k, l) = 0.75 * b.Q0(i, j, k, l) + 0.25 * b.Q(i, j, k, l) +
                          0.25 * b.dQ(i, j, k, l) * dt;
      });
}

void rk3s3(block_ &b, const double &dt) {
  //-------------------------------------------------------------------------------------------|
  // Apply RK3 stage 3
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "rk3 stage 3", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.Q(i, j, k, l) = (b.Q0(i, j, k, l) + 2.0 * b.Q(i, j, k, l) +
                           2.0 * b.dQ(i, j, k, l) * dt) /
                          3.0;
      });
}
