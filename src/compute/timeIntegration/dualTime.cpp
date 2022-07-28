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

void DTrk2s2(block_ b, const double dt) {
  //-------------------------------------------------------------------------------------------|
  // Apply RK2 stage 2
  //-------------------------------------------------------------------------------------------|
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
  double rP, rU, rV, rW, rT, rYi;
  std::vector<double> returnResid;
  for (int l = 0; l < ne; l++) {
    returnResid.push_back(0.0);
  }

  // First we will find the max L_inf residual for each p,u,v,w,T primative
  // RECALL: Q0 array is actually primatives when using dual time
  for (block_ b : mb) {

    MDRange3 range_cc({b.ng, b.ng, b.ng},
                      {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});
    Kokkos::parallel_reduce(
        "residual", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k, double &rP,
                      double &rU, double &rV, double &rW, double &rT) {
          rP = abs(b.q(i, j, k, 0) - b.Q0(i, j, k, 0));
          rU = abs(b.q(i, j, k, 1) - b.Q0(i, j, k, 1));
          rV = abs(b.q(i, j, k, 2) - b.Q0(i, j, k, 2));
          rW = abs(b.q(i, j, k, 3) - b.Q0(i, j, k, 3));
          rT = abs(b.q(i, j, k, 4) - b.Q0(i, j, k, 4));
        },
        Kokkos::Max<double>(rP), Kokkos::Max<double>(rU),
        Kokkos::Max<double>(rV), Kokkos::Max<double>(rW),
        Kokkos::Max<double>(rT));

    returnResid[0] = fmax(rP, returnResid[0]);
    returnResid[1] = fmax(rU, returnResid[1]);
    returnResid[2] = fmax(rV, returnResid[2]);
    returnResid[3] = fmax(rW, returnResid[3]);
    returnResid[4] = fmax(rT, returnResid[4]);

    for (int n = 5; n < ne; n++) {
      Kokkos::parallel_reduce(
          "residual", range_cc,
          KOKKOS_LAMBDA(const int i, const int j, const int k, double &rYi) {
            rYi = abs(b.q(i, j, k, n) - b.Q0(i, j, k, n));
          },
          Kokkos::Max<double>(rYi));
      returnResid[n] = fmax(rP, returnResid[n]);
    }
  }

  return returnResid;
}
