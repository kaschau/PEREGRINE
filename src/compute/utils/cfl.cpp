#include "Kokkos_Core.hpp"
#include "array"
#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include "vector"

std::array<double, 3> CFLmax(const std::vector<block_> &mb) {

  //-------------------------------------------------------------------------------------------|
  // Compute the max acoustic and convective CFL factor speed/dx
  //-------------------------------------------------------------------------------------------|
  double CFLmaxA, CFLmaxC, CFLmaxR;
  double returnMaxA = 0.0;
  double returnMaxC = 0.0;
  double returnMaxR = 0.0;

  for (const block_ b : mb) {
    double iMult = 1.0;
    double jMult = 1.0;
    double kMult = 1.0;
    if (b.ni == 2) {
      iMult = 0.0;
    }
    if (b.nj == 2) {
      jMult = 0.0;
    }
    if (b.nk == 2) {
      kMult = 0.0;
    }
    MDRange3 range_cc({b.ng, b.ng, b.ng},
                      {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});
    Kokkos::parallel_reduce(
        "CFLmax", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k, double &CFLA,
                      double &CFLC, double &CFLR) {
          // Cell lengths
          double dI = sqrt(pow(b.ixc(i + 1, j, k) - b.ixc(i, j, k), 2.0) +
                           pow(b.iyc(i + 1, j, k) - b.iyc(i, j, k), 2.0) +
                           pow(b.izc(i + 1, j, k) - b.izc(i, j, k), 2.0));
          double dJ = sqrt(pow(b.jxc(i, j + 1, k) - b.jxc(i, j, k), 2.0) +
                           pow(b.jyc(i, j + 1, k) - b.jyc(i, j, k), 2.0) +
                           pow(b.jzc(i, j + 1, k) - b.jzc(i, j, k), 2.0));
          double dK = sqrt(pow(b.kxc(i, j, k + 1) - b.kxc(i, j, k), 2.0) +
                           pow(b.kyc(i, j, k + 1) - b.kyc(i, j, k), 2.0) +
                           pow(b.kzc(i, j, k + 1) - b.kzc(i, j, k), 2.0));

          // Find max convective CFL
          double &u = b.q(i, j, k, 1);
          double &v = b.q(i, j, k, 2);
          double &w = b.q(i, j, k, 3);

          double uI =
              sqrt(pow(0.5 * (b.inx(i, j, k) + b.inx(i + 1, j, k)) * u, 2.0) +
                   pow(0.5 * (b.iny(i, j, k) + b.iny(i + 1, j, k)) * v, 2.0) +
                   pow(0.5 * (b.inz(i, j, k) + b.inz(i + 1, j, k)) * w, 2.0));
          double uJ =
              sqrt(pow(0.5 * (b.jnx(i, j, k) + b.jnx(i, j + 1, k)) * u, 2.0) +
                   pow(0.5 * (b.jny(i, j, k) + b.jny(i, j + 1, k)) * v, 2.0) +
                   pow(0.5 * (b.jnz(i, j, k) + b.jnz(i, j + 1, k)) * w, 2.0));
          double uK =
              sqrt(pow(0.5 * (b.knx(i, j, k) + b.knx(i, j, k + 1)) * u, 2.0) +
                   pow(0.5 * (b.kny(i, j, k) + b.kny(i, j, k + 1)) * v, 2.0) +
                   pow(0.5 * (b.knz(i, j, k) + b.knz(i, j, k + 1)) * w, 2.0));

          double &c = b.qh(i, j, k, 3);

          // i mult
          CFLA = fmax(CFLA, iMult * c / dI);
          CFLC = fmax(CFLC, iMult * uI / dI);
          CFLR = fmax(CFLR, iMult * (uI + c) / dI);
          // j mult
          CFLA = fmax(CFLA, jMult * c / dJ);
          CFLC = fmax(CFLC, jMult * uJ / dJ);
          CFLR = fmax(CFLR, jMult * (uJ + c) / dJ);
          // k mult
          CFLA = fmax(CFLA, kMult * c / dK);
          CFLC = fmax(CFLC, kMult * uK / dK);
          CFLR = fmax(CFLR, kMult * (uK + c) / dK);
        },
        Kokkos::Max<double>(CFLmaxA), Kokkos::Max<double>(CFLmaxC),
        Kokkos::Max<double>(CFLmaxR));
    returnMaxA = fmax(CFLmaxA, returnMaxA);
    returnMaxC = fmax(fmax(CFLmaxC, returnMaxC), 1e-16);
    returnMaxR = fmax(CFLmaxR, returnMaxR);
  }

  return {returnMaxA, returnMaxC, returnMaxR};
}
