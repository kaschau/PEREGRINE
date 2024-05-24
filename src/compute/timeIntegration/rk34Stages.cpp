#include "block_.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

void rk34s1(block_ &b, const double &dt) {
  //-------------------------------------------------------------------------------------------|
  // Apply RK34 stage 1
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "rk34 stage 1", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        // store zeroth stage
        b.Q0(i, j, k, l) = b.Q(i, j, k, l);
        b.Q(i, j, k, l) = b.Q(i, j, k, l) + 0.5 * dt * b.dQ(i, j, k, l);
      });
}

void rk34s2(block_ &b, const double &dt) {
  //-------------------------------------------------------------------------------------------|
  // Apply RK34 stage 2
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "rk34 stage 2", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.Q(i, j, k, l) = b.Q(i, j, k, l) + 0.5 * dt * b.dQ(i, j, k, l);
      });
}

void rk34s3(block_ &b, const double &dt) {
  //-------------------------------------------------------------------------------------------|
  // Apply RK34 stage 3
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "rk34 stage 3", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.Q(i, j, k, l) = 2.0 / 3.0 * b.Q0(i, j, k, l) +
                          1.0 / 3.0 * b.Q(i, j, k, l) +
                          1.0 / 6.0 * b.dQ(i, j, k, l) * dt;
      });
}

void rk34s4(block_ &b, const double &dt) {
  //-------------------------------------------------------------------------------------------|
  // Apply RK34 stage 4
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "rk34 stage 4", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.Q(i, j, k, l) = b.Q(i, j, k, l) + 0.5 * b.dQ(i, j, k, l) * dt;
      });
}
