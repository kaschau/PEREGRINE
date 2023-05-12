#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "math.h"

double computeEntropy(const std::vector<block_> &mb) {

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
          double &cp = b.qh(i, j, k, 1);
          double &gamma = b.qh(i, j, k, 0);
          double &T = b.q(i, j, k, 4);
          double &rho = b.Q(i, j, k, 0);
          double cv = cp / gamma;
          double R = cp - cv;

          s += rho * (cv * log(T) - R * log(rho));
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
