#include "kernel.hpp"
#include <numeric>

PG_RANGE(cellCenters)
struct jamesonPressure {
  cellCenterOut phi;
  cellCenterIn q;
  dims d;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const int ni = d->ni, nj = d->nj, nk = d->nk;

    const double &p = q(0);

    const double &pip = q(+I, 0);
    const double &pim = q(-I, 0);

    const double &pjp = q(+J, 0);
    const double &pjm = q(-J, 0);

    const double &pkp = q(+K, 0);
    const double &pkm = q(-K, 0);

    double ri = abs(pip - 2.0 * p + pim) / abs(pip + 2.0 * p + pim);
    phi(0) = ri;

    double rj = abs(pjp - 2.0 * p + pjm) / abs(pjp + 2.0 * p + pjm);
    phi(1) = rj;

    double rk = abs(pkp - 2.0 * p + pkm) / abs(pkp + 2.0 * p + pkm);
    phi(2) = rk;
  }
};

PG_ABI void pgJamesonPressure(const jamesonPressure &k, const pgTiling &t) {
  forCells("Compute switch from pressure", t, k);
}
