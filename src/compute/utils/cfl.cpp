#include "array"
#include "block_.hpp"
#include "compute.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include "vector"
#include <Kokkos_Core.hpp>

std::array<double, 3> CFLmax(const std::vector<block_> &mb) {

  //-------------------------------------------------------------------------------------------|
  // Compute the max acoustic and convective CFL factor speed/dx
  //-------------------------------------------------------------------------------------------|
  double CFLmaxA, CFLmaxC, CFLmaxR;
  double returnMaxA = 0.0;
  double returnMaxC = 0.0;
  double returnMaxR = 0.0;

  for (const block_ &b : mb) {
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
          const double &dI = b.dIJK(i, j, k, 0);
          const double &dJ = b.dIJK(i, j, k, 1);
          const double &dK = b.dIJK(i, j, k, 2);

          // Find max convective CFL
          double S0, S1;
          double inx0, iny0, inz0, inx1, iny1, inz1;
          faceNormal(b.iS(i, j, k, 0), b.iS(i, j, k, 1), b.iS(i, j, k, 2), S0,
                     inx0, iny0, inz0);
          faceNormal(b.iS(i + 1, j, k, 0), b.iS(i + 1, j, k, 1),
                     b.iS(i + 1, j, k, 2), S1, inx1, iny1, inz1);
          double jnx0, jny0, jnz0, jnx1, jny1, jnz1;
          faceNormal(b.jS(i, j, k, 0), b.jS(i, j, k, 1), b.jS(i, j, k, 2), S0,
                     jnx0, jny0, jnz0);
          faceNormal(b.jS(i, j + 1, k, 0), b.jS(i, j + 1, k, 1),
                     b.jS(i, j + 1, k, 2), S1, jnx1, jny1, jnz1);
          double knx0, kny0, knz0, knx1, kny1, knz1;
          faceNormal(b.kS(i, j, k, 0), b.kS(i, j, k, 1), b.kS(i, j, k, 2), S0,
                     knx0, kny0, knz0);
          faceNormal(b.kS(i, j, k + 1, 0), b.kS(i, j, k + 1, 1),
                     b.kS(i, j, k + 1, 2), S1, knx1, kny1, knz1);
          double &u = b.q(i, j, k, 1);
          double &v = b.q(i, j, k, 2);
          double &w = b.q(i, j, k, 3);

          double uI = sqrt(pow(0.5 * (inx0 + inx1) * u, 2.0) +
                           pow(0.5 * (iny0 + iny1) * v, 2.0) +
                           pow(0.5 * (inz0 + inz1) * w, 2.0));
          double uJ = sqrt(pow(0.5 * (jnx0 + jnx1) * u, 2.0) +
                           pow(0.5 * (jny0 + jny1) * v, 2.0) +
                           pow(0.5 * (jnz0 + jnz1) * w, 2.0));
          double uK = sqrt(pow(0.5 * (knx0 + knx1) * u, 2.0) +
                           pow(0.5 * (kny0 + kny1) * v, 2.0) +
                           pow(0.5 * (knz0 + knz1) * w, 2.0));

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
