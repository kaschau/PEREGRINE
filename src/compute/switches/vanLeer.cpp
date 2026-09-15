#include "kernel.hpp"
#include <numeric>

PG_RANGE(interior)
struct vanLeer {
  out phi;
  in q;
  dims d;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const int ni = d->ni, nj = d->nj, nk = d->nk;

    double eps = 0.001;

    const double &p = q(0);

    const double &pip = q(+I, 0);
    const double &pim = q(-I, 0);

    const double &pjp = q(+J, 0);
    const double &pjm = q(-J, 0);

    const double &pkp = q(+K, 0);
    const double &pkm = q(-K, 0);

    double ri = abs(pip - 2.0 * p + pim) /
                ((1.0 - eps) * (abs(pip - p) + abs(p - pim)) +
                 eps * (pip + 2.0 * p + pim));
    phi(0) = ri;

    double rj = abs(pjp - 2.0 * p + pjm) /
                ((1.0 - eps) * (abs(pjp - p) + abs(p - pjm)) +
                 eps * (pjp + 2.0 * p + pjm));
    phi(1) = rj;

    double rk = abs(pkp - 2.0 * p + pkm) /
                ((1.0 - eps) * (abs(pkp - p) + abs(p - pkm)) +
                 eps * (pkp + 2.0 * p + pkm));
    phi(2) = rk;
  }
};

PG_ABI void pgVanLeer(const vanLeer &k, const pgTiling &t) {
  forCells("Compute switch from pressure", t, k);
}
