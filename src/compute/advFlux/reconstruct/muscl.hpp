// Second order MUSCL: each side's state at the face is its cell's, moved
// half a cell along a limited slope. The density, velocity and internal
// energy are reconstructed; the pressure and sound speed take the energy's
// limiter, the species the density's, and the total energy is rebuilt from
// what was reconstructed. The limiter is PG_LIMITER, forced in ahead by
// the jit.
#ifndef __reconstructMuscl_H__
#define __reconstructMuscl_H__

#include "advFlux/faceState.hpp"
#include "faces.hpp"

PG_STENCIL(2);

struct muscl {
  cellStradVecIn Q, q, qh;
  faceVecOut F;
  faceVecIn A;

  // the limiter's phi of the slope ratio, guarding the flat side: a slope
  // against none is the limiter's limit, none against none is none
  static KOKKOS_INLINE_FUNCTION fpdtype phi(fpdtype dm, fpdtype dp) {
    if (dp == 0.0)
      return dm == 0.0 ? 0.0 : PG_LIMITER::limit;
    return PG_LIMITER::phi(dm / dp);
  }
  // one variable at the four cells: the limiter either side of the face,
  // and the face values they give
  struct slope {
    fpdtype phiL, phiR;
    KOKKOS_INLINE_FUNCTION slope(fpdtype vLL, fpdtype vL, fpdtype vR,
                                 fpdtype vRR)
        : phiL(phi(vL - vLL, vR - vL)), phiR(phi(vR - vL, vRR - vR)) {}
    KOKKOS_INLINE_FUNCTION fpdtype left(fpdtype vL, fpdtype vR) const {
      return vL + 0.5 * phiL * (vR - vL);
    }
    KOKKOS_INLINE_FUNCTION fpdtype right(fpdtype vR, fpdtype vRR) const {
      return vR - 0.5 * phiR * (vRR - vR);
    }
  };
  struct sides {
    faceState L, R;
    // the density's slope, which the species take
    slope rho;
  };

  KOKKOS_INLINE_FUNCTION sides states() const {
    const fpdtype rhoLL = Q.LL(0), rhoL = Q.L(0), rhoR = Q.R(0),
                  rhoRR = Q.RR(0);
    sides s{{}, {}, slope(rhoLL, rhoL, rhoR, rhoRR)};
    faceState &L = s.L, &R = s.R;
    L.rho = s.rho.left(rhoL, rhoR);
    R.rho = s.rho.right(rhoR, rhoRR);
    fpdtype *uL[] = {&L.u, &L.v, &L.w}, *uR[] = {&R.u, &R.v, &R.w};
    for (int d = 0; d < 3; d++) {
      const fpdtype vLL = Q.LL(1 + d) / rhoLL, vL = Q.L(1 + d) / rhoL,
                    vR = Q.R(1 + d) / rhoR, vRR = Q.RR(1 + d) / rhoRR;
      const slope u(vLL, vL, vR, vRR);
      *uL[d] = u.left(vL, vR);
      *uR[d] = u.right(vR, vRR);
    }
    // the internal energy per mass, whose limiter p and c take too
    const slope e(qh.LL(4) / rhoLL, qh.L(4) / rhoL, qh.R(4) / rhoR,
                  qh.RR(4) / rhoRR);
    const fpdtype eL = e.left(qh.L(4) / rhoL, qh.R(4) / rhoR),
                  eR = e.right(qh.R(4) / rhoR, qh.RR(4) / rhoRR);
    L.p = e.left(q.L(0), q.R(0));
    R.p = e.right(q.R(0), q.RR(0));
    L.c = e.left(qh.L(3), qh.R(3));
    R.c = e.right(qh.R(3), qh.RR(3));
    L.rhou = L.rho * L.u, L.rhov = L.rho * L.v, L.rhow = L.rho * L.w;
    R.rhou = R.rho * R.u, R.rhov = R.rho * R.v, R.rhow = R.rho * R.w;
    L.E = L.rho * (eL + 0.5 * (L.u * L.u + L.v * L.v + L.w * L.w));
    R.E = R.rho * (eR + 0.5 * (R.u * R.u + R.v * R.v + R.w * R.w));
    return s;
  }
  KOKKOS_INLINE_FUNCTION void species(const sides &s, int n, fpdtype &L,
                                      fpdtype &R) const {
    L = s.rho.left(Q.L(5 + n), Q.R(5 + n));
    R = s.rho.right(Q.R(5 + n), Q.RR(5 + n));
  }
};

#endif
