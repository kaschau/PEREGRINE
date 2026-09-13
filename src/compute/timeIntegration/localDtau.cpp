#include "array"
#include "dualTime.hpp"
#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include "vector"
#include <Kokkos_Core.hpp>

PG_ABI void pgLocalDtau(int count, pgIn *Q_, pgIn *dIJK_, pgOut *dtau_,
                        pgIn *iS_, pgIn *jS_, pgIn *kS_, pgIn *q_, pgIn *qh_,
                        pgIn *qt_, const pgDims *d, bool viscous) {
  for (int e = 0; e < count; e++) {
    auto Q = as4(Q_[e]);
    auto dIJK = as4(dIJK_[e]);
    auto dtau = as3(dtau_[e]);
    auto iS = as4(iS_[e]);
    auto jS = as4(jS_[e]);
    auto kS = as4(kS_[e]);
    auto q = as4(q_[e]);
    auto qh = as4(qh_[e]);
    auto qt = as4(qt_[e]);
    const int ni = d[e].ni, nj = d[e].nj, nk = d[e].nk;
    //-------------------------------------------------------------------------------------------|
    // Compute local pseudo time step
    //-------------------------------------------------------------------------------------------|
    double iMult = 1.0;
    double jMult = 1.0;
    double kMult = 1.0;
    if (ni == 2) {
      iMult = Kokkos::Experimental::infinity<double>::value;
    }
    if (nj == 2) {
      jMult = Kokkos::Experimental::infinity<double>::value;
    }
    if (nk == 2) {
      kMult = Kokkos::Experimental::infinity<double>::value;
    }

    MDRange3 range_cc({ng, ng, ng}, {ni + ng - 1, nj + ng - 1, nk + ng - 1});
    Kokkos::parallel_for(
        "localDtau", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          // Cell lengths
          const double &dI = dIJK(i, j, k, 0);
          const double &dJ = dIJK(i, j, k, 1);
          const double &dK = dIJK(i, j, k, 2);

          // Find max convective CFL
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

          double uI = sqrt(pow(0.5 * (inx0 + inx1) * u, 2.0) +
                           pow(0.5 * (iny0 + iny1) * v, 2.0) +
                           pow(0.5 * (inz0 + inz1) * w, 2.0));
          double uJ = sqrt(pow(0.5 * (jnx0 + jnx1) * u, 2.0) +
                           pow(0.5 * (jny0 + jny1) * v, 2.0) +
                           pow(0.5 * (jnz0 + jnz1) * w, 2.0));
          double uK = sqrt(pow(0.5 * (knx0 + knx1) * u, 2.0) +
                           pow(0.5 * (kny0 + kny1) * v, 2.0) +
                           pow(0.5 * (knz0 + knz1) * w, 2.0));

          const double &c = qh(i, j, k, 3);

          double pseudoCFL = 0.5;
          double pseudoVNN = 0.1;

          // the preconditioned system's wave speeds set the pseudo step
          const double nu = viscous ? qt(i, j, k, 0) / Q(i, j, k, 0) : 0.0;
          const double Ur =
              referenceVelocity(sqrt(u * u + v * v + w * w), c, nu, iMult * dI,
                                jMult * dJ, kMult * dK);
          // the preconditioned system propagates u' +- c', not u + c
          const double alpha = 0.5 * (1.0 - Ur * Ur / (c * c));
          const double a2 = alpha * alpha;
          const double Ur2 = Ur * Ur;

          double dtauCell = Kokkos::Experimental::infinity<double>::value;
          dtauCell = fmin(dtauCell, iMult * pseudoCFL * dI /
                                        (abs((1.0 - alpha) * uI) +
                                         sqrt(a2 * uI * uI + Ur2)));
          dtauCell = fmin(dtauCell, jMult * pseudoCFL * dJ /
                                        (abs((1.0 - alpha) * uJ) +
                                         sqrt(a2 * uJ * uJ + Ur2)));
          dtauCell = fmin(dtauCell, kMult * pseudoCFL * dK /
                                        (abs((1.0 - alpha) * uK) +
                                         sqrt(a2 * uK * uK + Ur2)));
          if (viscous) {
            dtauCell = fmin(dtauCell, iMult * pseudoVNN * pow(dI, 2.0) / nu);
            dtauCell = fmin(dtauCell, jMult * pseudoVNN * pow(dJ, 2.0) / nu);
            dtauCell = fmin(dtauCell, kMult * pseudoVNN * pow(dK, 2.0) / nu);
          }

          dtau(i, j, k) = dtauCell;
        });
  }
}
