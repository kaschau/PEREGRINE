#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

PG_ABI void pgRk4s1(const pgView *Q_, const pgView *Q0_, const pgView *Q1_,
                    const pgView *dQ_, const pgDims *d, double dt) {
  auto Q = as4(*Q_);
  auto Q0 = as4(*Q0_);
  auto Q1 = as4(*Q1_);
  auto dQ = as4(*dQ_);
  const int ni = d->ni, nj = d->nj, nk = d->nk;
  //-------------------------------------------------------------------------------------------|
  // Apply RK4 stage 1
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({ng, ng, ng, 0},
                    {ni + ng - 1, nj + ng - 1, nk + ng - 1, ne});
  Kokkos::parallel_for(
      "rk4 stage 1", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        // store zeroth stage
        Q0(i, j, k, l) = Q(i, j, k, l);
        Q1(i, j, k, l) = dt * dQ(i, j, k, l);
        Q(i, j, k, l) = Q0(i, j, k, l) + 0.5 * Q1(i, j, k, l);
      });
}

PG_ABI void pgRk4s2(const pgView *Q_, const pgView *Q0_, const pgView *Q2_,
                    const pgView *dQ_, const pgDims *d, double dt) {
  auto Q = as4(*Q_);
  auto Q0 = as4(*Q0_);
  auto Q2 = as4(*Q2_);
  auto dQ = as4(*dQ_);
  const int ni = d->ni, nj = d->nj, nk = d->nk;
  //-------------------------------------------------------------------------------------------|
  // Apply RK4 stage 2
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({ng, ng, ng, 0},
                    {ni + ng - 1, nj + ng - 1, nk + ng - 1, ne});
  Kokkos::parallel_for(
      "rk4 stage 2", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        Q2(i, j, k, l) = dt * dQ(i, j, k, l);
        Q(i, j, k, l) = Q0(i, j, k, l) + 0.5 * Q2(i, j, k, l);
      });
}

PG_ABI void pgRk4s3(const pgView *Q_, const pgView *Q0_, const pgView *Q3_,
                    const pgView *dQ_, const pgDims *d, double dt) {
  auto Q = as4(*Q_);
  auto Q0 = as4(*Q0_);
  auto Q3 = as4(*Q3_);
  auto dQ = as4(*dQ_);
  const int ni = d->ni, nj = d->nj, nk = d->nk;
  //-------------------------------------------------------------------------------------------|
  // Apply RK4 stage 3
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({ng, ng, ng, 0},
                    {ni + ng - 1, nj + ng - 1, nk + ng - 1, ne});
  Kokkos::parallel_for(
      "rk4 stage 3", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        Q3(i, j, k, l) = dt * dQ(i, j, k, l);
        Q(i, j, k, l) = Q0(i, j, k, l) + Q3(i, j, k, l);
      });
}

PG_ABI void pgRk4s4(const pgView *Q_, const pgView *Q0_, const pgView *Q1_,
                    const pgView *Q2_, const pgView *Q3_, const pgView *dQ_,
                    const pgDims *d, double dt) {
  auto Q = as4(*Q_);
  auto Q0 = as4(*Q0_);
  auto Q1 = as4(*Q1_);
  auto Q2 = as4(*Q2_);
  auto Q3 = as4(*Q3_);
  auto dQ = as4(*dQ_);
  const int ni = d->ni, nj = d->nj, nk = d->nk;
  //-------------------------------------------------------------------------------------------|
  // Apply RK4 stage 4
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({ng, ng, ng, 0},
                    {ni + ng - 1, nj + ng - 1, nk + ng - 1, ne});
  Kokkos::parallel_for(
      "rk4 stage 4", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        Q(i, j, k, l) =
            Q0(i, j, k, l) + (Q1(i, j, k, l) + 2.0 * Q2(i, j, k, l) +
                              2.0 * Q3(i, j, k, l) + dt * dQ(i, j, k, l)) /
                                 6.0;
      });
}
