#include "kernel.hpp"

// The rank's max acoustic, convective and combined CFL speeds (speed/dx),
// into cfl[3]; python combines the ranks.

// the three speeds, reduced together
struct cfl3 {
  fpdtype a, c, r;
};
struct maxCfl {
  using reducer = maxCfl;
  using value_type = cfl3;
  using result_view_type =
      Kokkos::View<cfl3 *, hostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  cfl3 &value;
  KOKKOS_INLINE_FUNCTION maxCfl(cfl3 &v) : value(v) {}
  KOKKOS_INLINE_FUNCTION void join(cfl3 &d, const cfl3 &s) const {
    d.a = fmax(d.a, s.a), d.c = fmax(d.c, s.c), d.r = fmax(d.r, s.r);
  }
  KOKKOS_INLINE_FUNCTION void init(cfl3 &v) const { v = {0.0, 0.0, 0.0}; }
  KOKKOS_INLINE_FUNCTION cfl3 &reference() const { return value; }
  KOKKOS_INLINE_FUNCTION result_view_type view() const {
    return result_view_type(&value, 1);
  }
  KOKKOS_INLINE_FUNCTION bool references_scalar() const { return true; }
};

PG_RANGE(cellCenters)
struct CFLmax {
  cellVecIn dIJK, Q, qh;
  iFaceStradVecIn iS;
  jFaceStradVecIn jS;
  kFaceStradVecIn kS;
  KOKKOS_INLINE_FUNCTION void operator()(cfl3 &m) const {
    fpdtype CFLA = m.a, CFLC = m.c, CFLR = m.r;
    // the cell lengths; an axis not marched in is infinitely long
    const fpdtype &dI = dIJK(0);
    const fpdtype &dJ = dIJK(1);
    const fpdtype &dK = dIJK(2);

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

    const fpdtype uI = sqrt(pow(0.5 * (inx0 + inx1) * u, 2.0) +
                            pow(0.5 * (iny0 + iny1) * v, 2.0) +
                            pow(0.5 * (inz0 + inz1) * w, 2.0));
    const fpdtype uJ = sqrt(pow(0.5 * (jnx0 + jnx1) * u, 2.0) +
                            pow(0.5 * (jny0 + jny1) * v, 2.0) +
                            pow(0.5 * (jnz0 + jnz1) * w, 2.0));
    const fpdtype uK = sqrt(pow(0.5 * (knx0 + knx1) * u, 2.0) +
                            pow(0.5 * (kny0 + kny1) * v, 2.0) +
                            pow(0.5 * (knz0 + knz1) * w, 2.0));
    const fpdtype &c = qh(3);

    CFLA = fmax(CFLA, c / dI);
    CFLC = fmax(CFLC, uI / dI);
    CFLR = fmax(CFLR, (uI + c) / dI);
    CFLA = fmax(CFLA, c / dJ);
    CFLC = fmax(CFLC, uJ / dJ);
    CFLR = fmax(CFLR, (uJ + c) / dJ);
    CFLA = fmax(CFLA, c / dK);
    CFLC = fmax(CFLC, uK / dK);
    CFLR = fmax(CFLR, (uK + c) / dK);
    m.a = CFLA, m.c = CFLC, m.r = CFLR;
  }
};

PG_ABI void pgCFLmax(const CFLmax &k, const pgTiling &t, fpdtype *cfl) {
  cfl3 total{0.0, 0.0, 0.0};
  reduceCells("CFLmax", t, k, maxCfl(total));
  cfl[0] = total.a, cfl[1] = total.c, cfl[2] = total.r;
}
