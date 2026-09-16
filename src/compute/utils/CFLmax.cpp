#include "kernel.hpp"

// The rank's max acoustic, convective and combined CFL speeds (speed/dx),
// into cfl[3]; python combines the ranks.

// the three speeds, reduced together
struct cfl3 {
  double a, c, r;
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
  cellCenterIn dIJK, iS, jS, kS, Q, qh;
  dims d;
  KOKKOS_INLINE_FUNCTION void operator()(cfl3 &m) const {
    // a direction one cell thick is not marched in
    const double iMult = d->ni == 2 ? 0.0 : 1.0;
    const double jMult = d->nj == 2 ? 0.0 : 1.0;
    const double kMult = d->nk == 2 ? 0.0 : 1.0;
    double CFLA = m.a, CFLC = m.c, CFLR = m.r;
    const double &dI = dIJK(0);
    const double &dJ = dIJK(1);
    const double &dK = dIJK(2);

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
    // the velocity off the conserved state
    const double rhoinv = 1.0 / Q(0);
    const double u = Q(1) * rhoinv;
    const double v = Q(2) * rhoinv;
    const double w = Q(3) * rhoinv;

    const double uI = sqrt(pow(0.5 * (inx0 + inx1) * u, 2.0) +
                           pow(0.5 * (iny0 + iny1) * v, 2.0) +
                           pow(0.5 * (inz0 + inz1) * w, 2.0));
    const double uJ = sqrt(pow(0.5 * (jnx0 + jnx1) * u, 2.0) +
                           pow(0.5 * (jny0 + jny1) * v, 2.0) +
                           pow(0.5 * (jnz0 + jnz1) * w, 2.0));
    const double uK = sqrt(pow(0.5 * (knx0 + knx1) * u, 2.0) +
                           pow(0.5 * (kny0 + kny1) * v, 2.0) +
                           pow(0.5 * (knz0 + knz1) * w, 2.0));
    const double &c = qh(3);

    CFLA = fmax(CFLA, iMult * c / dI);
    CFLC = fmax(CFLC, iMult * uI / dI);
    CFLR = fmax(CFLR, iMult * (uI + c) / dI);
    CFLA = fmax(CFLA, jMult * c / dJ);
    CFLC = fmax(CFLC, jMult * uJ / dJ);
    CFLR = fmax(CFLR, jMult * (uJ + c) / dJ);
    CFLA = fmax(CFLA, kMult * c / dK);
    CFLC = fmax(CFLC, kMult * uK / dK);
    CFLR = fmax(CFLR, kMult * (uK + c) / dK);
    m.a = CFLA, m.c = CFLC, m.r = CFLR;
  }
};

PG_ABI void pgCFLmax(const CFLmax &k, const pgTiling &t, double *cfl) {
  cfl3 total{0.0, 0.0, 0.0};
  reduceCells("CFLmax", t, k, maxCfl(total));
  cfl[0] = total.a, cfl[1] = total.c, cfl[2] = total.r;
}
