#include "Kokkos_Core.hpp"
#include "array"
#include "block_.hpp"
#include "kokkos_types.hpp"
#include "math.h"
#include "vector"

void dQdt(block_ b, const double dt) {
  //-------------------------------------------------------------------------------------------|
  // Start off dQ with real time derivative source term
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "dQdt", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.dQ(i, j, k, l) = (3.0 * b.Q(i, j, k, l) - 4.0 * b.Qn(i, j, k, l) +
                            b.Qnm1(i, j, k, l)) /
                           (2 * dt);
      });
}

void DTrk2s1(block_ b, const double dt) {
  //-------------------------------------------------------------------------------------------|
  // Apply RK2 stage 1
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "Dual time rk2 stage 1", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.q(i, j, k, l) += b.dQ(i, j, k, l) * dt;
      });
}

//-------------------------------------------------------------------------------------------|
// Apply RK2 stage 2
//-------------------------------------------------------------------------------------------|
void DTrk2s2(block_ b, const double dt) {
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "Dual TIme rk2 stage 2", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        b.q(i, j, k, l) = 0.5 * b.Q0(i, j, k, l) +
                          0.5 * (b.q(i, j, k, l) + dt * b.dQ(i, j, k, l));
      });
}

void invertDQ(block_ b, const double dt, const double dtau) {
  //-------------------------------------------------------------------------------------------|
  // Invert \Gamma dq = dQ to solver for dqdt
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_cc({b.ng, b.ng, b.ng},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});
  Kokkos::parallel_for("dq = \\Gamma^{-1} dQ", range_cc,
                       KOKKOS_LAMBDA(const int i, const int j, const int k){

                           // Do your thing...

                       });
}

std::vector<double> residual(std::vector<block_> mb) {
  //-------------------------------------------------------------------------------------------|
  // Compute max residual for each primative
  //-------------------------------------------------------------------------------------------|

  int ne = mb[0].ne;
  std::vector<double> resid;
  std::vector<double> testResid;
  for (int l = 0; l < ne; l++) {
    resid.pop_back();
    testResid.pop_back();
    resid[l] = 0.0;
    testResid[l] = 0.0;
  }

  for (block_ b : mb) {

    //   MDRange3 range_cc({b.ng, b.ng, b.ng},
    //                     {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});
    //   Kokkos::parallel_reduce(
    //       "residual", range_cc,
    //       KOKKOS_LAMBDA(const int i, const int j, const int k, double &CFLA,
    //                     double &CFLC, double &CFLR){

    //       },

    //       Kokkos::Max<double>(CFLmaxA), Kokkos::Max<double>(CFLmaxC),
    //       Kokkos::Max<double>(CFLmaxR));

    //   returnMaxA = fmax(CFLmaxA, returnMaxA);
    //   returnMaxC = fmax(fmax(CFLmaxC, returnMaxC), 1e-16);
    //   returnMaxR = fmax(CFLmaxR, returnMaxR);
  }

  return resid;
}
