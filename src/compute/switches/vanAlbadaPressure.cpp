#include "kernel.hpp"
#include <numeric>

PG_RANGE(interior)
struct vanAlbadaPressure {
  out phi;
  in q;
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

    double ri = (p - pim + 1e-16) / (pip - p + 1e-16);
    phi(0) = 1.0 - (ri + abs(ri)) / (1.0 + pow(ri, 2.0));

    double rj = (p - pjm + 1e-16) / (pjp - p + 1e-16);
    phi(1) = 1.0 - (rj + abs(rj)) / (1.0 + pow(rj, 2.0));

    double rk = (p - pkm + 1e-16) / (pkp - p + 1e-16);
    phi(2) = 1.0 - (rk + abs(rk)) / (1.0 + pow(rk, 2.0));
  }
};

PG_ABI void pgVanAlbadaPressure(const vanAlbadaPressure &k, const pgTiling &t) {
  forCells("Compute switch from entropy", t, k);
}
