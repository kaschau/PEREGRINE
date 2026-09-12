#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <math.h>

// The reductions, each over every block of the rank; python combines the
// ranks.

// 0 if any conserved quantity in the interior is not finite

// into cfl[3]
PG_ABI void pgCFLmax(int count, const pgView *dIJK_, const pgView *iS_,
                     const pgView *jS_, const pgView *kS_, const pgView *q_,
                     const pgView *qh_, const pgDims *d, double *cfl) {
  cfl[0] = cfl[1] = cfl[2] = 0.0;
  for (int e = 0; e < count; e++) {
    auto dIJK = as4(dIJK_[e]), iS = as4(iS_[e]), jS = as4(jS_[e]),
         kS = as4(kS_[e]), q = as4(q_[e]), qh = as4(qh_[e]);
    const int ni = d[e].ni, nj = d[e].nj, nk = d[e].nk;
    // a direction one cell thick is not marched in
    const double iMult = ni == 2 ? 0.0 : 1.0;
    const double jMult = nj == 2 ? 0.0 : 1.0;
    const double kMult = nk == 2 ? 0.0 : 1.0;
    double CFLmaxA, CFLmaxC, CFLmaxR;
    MDRange3 range_cc({ng, ng, ng}, {ni + ng - 1, nj + ng - 1, nk + ng - 1});
    Kokkos::parallel_reduce(
        "CFLmax", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k, double &CFLA,
                      double &CFLC, double &CFLR) {
          const double &dI = dIJK(i, j, k, 0);
          const double &dJ = dIJK(i, j, k, 1);
          const double &dK = dIJK(i, j, k, 2);

          double S0, S1;
          double inx0, iny0, inz0, inx1, iny1, inz1;
          faceNormal(iS(i, j, k, 0), iS(i, j, k, 1), iS(i, j, k, 2), S0, inx0,
                     iny0, inz0);
          faceNormal(iS(i + 1, j, k, 0), iS(i + 1, j, k, 1), iS(i + 1, j, k, 2),
                     S1, inx1, iny1, inz1);
          double jnx0, jny0, jnz0, jnx1, jny1, jnz1;
          faceNormal(jS(i, j, k, 0), jS(i, j, k, 1), jS(i, j, k, 2), S0, jnx0,
                     jny0, jnz0);
          faceNormal(jS(i, j + 1, k, 0), jS(i, j + 1, k, 1), jS(i, j + 1, k, 2),
                     S1, jnx1, jny1, jnz1);
          double knx0, kny0, knz0, knx1, kny1, knz1;
          faceNormal(kS(i, j, k, 0), kS(i, j, k, 1), kS(i, j, k, 2), S0, knx0,
                     kny0, knz0);
          faceNormal(kS(i, j, k + 1, 0), kS(i, j, k + 1, 1), kS(i, j, k + 1, 2),
                     S1, knx1, kny1, knz1);
          const double &u = q(i, j, k, 1);
          const double &v = q(i, j, k, 2);
          const double &w = q(i, j, k, 3);

          const double uI = sqrt(pow(0.5 * (inx0 + inx1) * u, 2.0) +
                                 pow(0.5 * (iny0 + iny1) * v, 2.0) +
                                 pow(0.5 * (inz0 + inz1) * w, 2.0));
          const double uJ = sqrt(pow(0.5 * (jnx0 + jnx1) * u, 2.0) +
                                 pow(0.5 * (jny0 + jny1) * v, 2.0) +
                                 pow(0.5 * (jnz0 + jnz1) * w, 2.0));
          const double uK = sqrt(pow(0.5 * (knx0 + knx1) * u, 2.0) +
                                 pow(0.5 * (kny0 + kny1) * v, 2.0) +
                                 pow(0.5 * (knz0 + knz1) * w, 2.0));
          const double &c = qh(i, j, k, 3);

          CFLA = fmax(CFLA, iMult * c / dI);
          CFLC = fmax(CFLC, iMult * uI / dI);
          CFLR = fmax(CFLR, iMult * (uI + c) / dI);
          CFLA = fmax(CFLA, jMult * c / dJ);
          CFLC = fmax(CFLC, jMult * uJ / dJ);
          CFLR = fmax(CFLR, jMult * (uJ + c) / dJ);
          CFLA = fmax(CFLA, kMult * c / dK);
          CFLC = fmax(CFLC, kMult * uK / dK);
          CFLR = fmax(CFLR, kMult * (uK + c) / dK);
        },
        Kokkos::Max<double>(CFLmaxA), Kokkos::Max<double>(CFLmaxC),
        Kokkos::Max<double>(CFLmaxR));
    cfl[0] = fmax(cfl[0], CFLmaxA);
    cfl[1] = fmax(cfl[1], CFLmaxC);
    cfl[2] = fmax(cfl[2], CFLmaxR);
  }
}

// the rank's max and sum-of-squares residual of every primitive between q
