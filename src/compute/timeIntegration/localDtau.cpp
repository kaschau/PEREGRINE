#include "array"
#include "dualTime.hpp"
#include "kernel.hpp"
#include "vector"

PG_RANGE(cellCenters)
struct localDtau {
  cellCenterIn Q, dIJK, iS, jS, kS, q, qh, qt;
  cellCenterOut dtau;
  dims d;
  bool viscous;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const int ni = d->ni, nj = d->nj, nk = d->nk;
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

    // Cell lengths
    const double &dI = dIJK(0);
    const double &dJ = dIJK(1);
    const double &dK = dIJK(2);

    // Find max convective CFL
    double S0, S1;
    double inx0, iny0, inz0, inx1, iny1, inz1;
    faceNormal(iS(0), iS(1), iS(2), S0, inx0, iny0, inz0);
    faceNormal(iS(+I, 0), iS(+I, 1), iS(+I, 2), S1, inx1, iny1, inz1);
    double jnx0, jny0, jnz0, jnx1, jny1, jnz1;
    faceNormal(jS(0), jS(1), jS(2), S0, jnx0, jny0, jnz0);
    faceNormal(jS(+J, 0), jS(+J, 1), jS(+J, 2), S1, jnx1, jny1, jnz1);
    double knx0, kny0, knz0, knx1, kny1, knz1;
    faceNormal(kS(0), kS(1), kS(2), S0, knx0, kny0, knz0);
    faceNormal(kS(+K, 0), kS(+K, 1), kS(+K, 2), S1, knx1, kny1, knz1);
    const double &u = q(1);
    const double &v = q(2);
    const double &w = q(3);

    double uI = sqrt(pow(0.5 * (inx0 + inx1) * u, 2.0) +
                     pow(0.5 * (iny0 + iny1) * v, 2.0) +
                     pow(0.5 * (inz0 + inz1) * w, 2.0));
    double uJ = sqrt(pow(0.5 * (jnx0 + jnx1) * u, 2.0) +
                     pow(0.5 * (jny0 + jny1) * v, 2.0) +
                     pow(0.5 * (jnz0 + jnz1) * w, 2.0));
    double uK = sqrt(pow(0.5 * (knx0 + knx1) * u, 2.0) +
                     pow(0.5 * (kny0 + kny1) * v, 2.0) +
                     pow(0.5 * (knz0 + knz1) * w, 2.0));

    const double &c = qh(3);

    double pseudoCFL = 0.5;
    double pseudoVNN = 0.1;

    // the preconditioned system's wave speeds set the pseudo step
    const double nu = viscous ? qt(0) / Q(0) : 0.0;
    const double Ur = referenceVelocity(sqrt(u * u + v * v + w * w), c, nu,
                                        iMult * dI, jMult * dJ, kMult * dK);
    // the preconditioned system propagates u' +- c', not u + c
    const double alpha = 0.5 * (1.0 - Ur * Ur / (c * c));
    const double a2 = alpha * alpha;
    const double Ur2 = Ur * Ur;

    double dtauCell = Kokkos::Experimental::infinity<double>::value;
    dtauCell = fmin(dtauCell,
                    iMult * pseudoCFL * dI /
                        (abs((1.0 - alpha) * uI) + sqrt(a2 * uI * uI + Ur2)));
    dtauCell = fmin(dtauCell,
                    jMult * pseudoCFL * dJ /
                        (abs((1.0 - alpha) * uJ) + sqrt(a2 * uJ * uJ + Ur2)));
    dtauCell = fmin(dtauCell,
                    kMult * pseudoCFL * dK /
                        (abs((1.0 - alpha) * uK) + sqrt(a2 * uK * uK + Ur2)));
    if (viscous) {
      dtauCell = fmin(dtauCell, iMult * pseudoVNN * pow(dI, 2.0) / nu);
      dtauCell = fmin(dtauCell, jMult * pseudoVNN * pow(dJ, 2.0) / nu);
      dtauCell = fmin(dtauCell, kMult * pseudoVNN * pow(dK, 2.0) / nu);
    }

    dtau() = dtauCell;
  }
};

PG_ABI void pgLocalDtau(const localDtau &k, const pgTiling &t) {
  forCells("localDtau", t, k);
}
