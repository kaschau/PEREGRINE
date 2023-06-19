#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include "thtrdat_.hpp"

#ifdef NSCOMPILE
#define ns NS
#endif

double computeEntropy(const std::vector<block_> &mb, thtrdat_ &th) {

  //-------------------------------------------------------------------------------------------|
  // Compute the max acoustic and convective CFL factor speed/dx
  //-------------------------------------------------------------------------------------------|
  double returnS = 0.0;
  double tempS;

  for (const block_ b : mb) {
    MDRange3 range_cc({b.ng, b.ng, b.ng},
                      {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});
    Kokkos::parallel_reduce(
        "compute entropy", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k, double &s) {
          // Find max convective CFL
          double Y[ns];
          Y[ns - 1] = 1.0;
          for (int n = 0; n < ns - 1; n++) {
            Y[n] = b.q(i, j, k, 5 + n);
            Y[ns - 1] -= Y[n];
          }
          double &T = b.q(i, j, k, 4);
          double &rho = b.Q(i, j, k, 0);
          for (int n = 0; n < ns; n++) {
            if (Y[n] == 0.0) {
              continue;
            } else {
              double cpk = th.cp0(n);
              double Rk = th.Ru / th.MW(n);
              double cvk = cpk - Rk;
              s += rho * Y[n] * (cvk * log(T) - Rk * log(rho * Y[n]));
            }
          }
        },
        Kokkos::Sum<double>(tempS));
    returnS += tempS;
  }

  return returnS;
}

double sumEntropy(const std::vector<block_> &mb) {

  //-------------------------------------------------------------------------------------------|
  // Compute the max acoustic and convective CFL factor speed/dx
  //-------------------------------------------------------------------------------------------|
  double returnS = 0.0;
  double tempS;

  for (const block_ b : mb) {
    MDRange3 range_cc({b.ng, b.ng, b.ng},
                      {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});
    Kokkos::parallel_reduce(
        "compute entropy", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k, double &s) {
          s += b.s1(i, j, k);
        },
        Kokkos::Sum<double>(tempS));
    returnS += tempS;
  }

  return returnS;
}
