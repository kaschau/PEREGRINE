#include "array"
#include "dualTime.hpp"
#include "kernel.hpp"
#include "vector"

PG_RANGE(cellCenters)
struct localDtau {
  cellVecIn Q, dIJK, qh, qt;
  iFaceStradVecIn iS;
  jFaceStradVecIn jS;
  kFaceStradVecIn kS;
  cellScalOut dtau;
  dims d;
  bool viscous;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const int ni = d->ni, nj = d->nj, nk = d->nk;
    //-------------------------------------------------------------------------------------------|
    // Compute local pseudo time step
    //-------------------------------------------------------------------------------------------|
    fpdtype iMult = 1.0;
    fpdtype jMult = 1.0;
    fpdtype kMult = 1.0;
    if (ni == 2) {
      iMult = Kokkos::Experimental::infinity<fpdtype>::value;
    }
    if (nj == 2) {
      jMult = Kokkos::Experimental::infinity<fpdtype>::value;
    }
    if (nk == 2) {
      kMult = Kokkos::Experimental::infinity<fpdtype>::value;
    }

    // Cell lengths
    const fpdtype &dI = dIJK(0);
    const fpdtype &dJ = dIJK(1);
    const fpdtype &dK = dIJK(2);

    // Find max convective CFL
    fpdtype S0, S1;
    fpdtype inx0, iny0, inz0, inx1, iny1, inz1;
    faceNormal(iS.L(0), iS.L(1), iS.L(2), S0, inx0, iny0, inz0);
    faceNormal(iS.R(0), iS.R(1), iS.R(2), S1, inx1, iny1, inz1);
    fpdtype jnx0, jny0, jnz0, jnx1, jny1, jnz1;
    faceNormal(jS.L(0), jS.L(1), jS.L(2), S0, jnx0, jny0, jnz0);
    faceNormal(jS.R(0), jS.R(1), jS.R(2), S1, jnx1, jny1, jnz1);
    fpdtype knx0, kny0, knz0, knx1, kny1, knz1;
    faceNormal(kS.L(0), kS.L(1), kS.L(2), S0, knx0, kny0, knz0);
    faceNormal(kS.R(0), kS.R(1), kS.R(2), S1, knx1, kny1, knz1);
    // the velocity off the conserved state
    const fpdtype rhoinv = 1.0 / Q(0);
    const fpdtype u = Q(1) * rhoinv;
    const fpdtype v = Q(2) * rhoinv;
    const fpdtype w = Q(3) * rhoinv;

    fpdtype uI = sqrt(pow(0.5 * (inx0 + inx1) * u, 2.0) +
                      pow(0.5 * (iny0 + iny1) * v, 2.0) +
                      pow(0.5 * (inz0 + inz1) * w, 2.0));
    fpdtype uJ = sqrt(pow(0.5 * (jnx0 + jnx1) * u, 2.0) +
                      pow(0.5 * (jny0 + jny1) * v, 2.0) +
                      pow(0.5 * (jnz0 + jnz1) * w, 2.0));
    fpdtype uK = sqrt(pow(0.5 * (knx0 + knx1) * u, 2.0) +
                      pow(0.5 * (kny0 + kny1) * v, 2.0) +
                      pow(0.5 * (knz0 + knz1) * w, 2.0));

    const fpdtype &c = qh(3);

    fpdtype pseudoCFL = 0.5;
    fpdtype pseudoVNN = 0.1;

    // the preconditioned system's wave speeds set the pseudo step
    const fpdtype nu = viscous ? qt(0) * rhoinv : 0.0;
    const fpdtype Ur = referenceVelocity(sqrt(u * u + v * v + w * w), c, nu,
                                         iMult * dI, jMult * dJ, kMult * dK);
    // the preconditioned system propagates u' +- c', not u + c
    const fpdtype alpha = 0.5 * (1.0 - Ur * Ur / (c * c));
    const fpdtype a2 = alpha * alpha;
    const fpdtype Ur2 = Ur * Ur;

    fpdtype dtauCell = Kokkos::Experimental::infinity<fpdtype>::value;
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

    dtau = dtauCell;
  }
};

PG_ABI void pgLocalDtau(const localDtau &k, const pgTiling &t) {
  forCells("localDtau", t, k);
}
