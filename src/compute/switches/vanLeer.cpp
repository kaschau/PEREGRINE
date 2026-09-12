#include "abi.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <math.h>
#include <numeric>

PG_ABI void pgVanLeer(const pgView *phi_, const pgView *q_, const pgDims *d) {
  auto phi = as4(*phi_);
  auto q = as4(*q_);
  const int ng = d->ng, ni = d->ni, nj = d->nj, nk = d->nk;

  MDRange3 range_cc({ng, ng, ng}, {ni + ng - 1, nj + ng - 1, nk + ng - 1});

  Kokkos::parallel_for(
      "Compute switch from pressure", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double eps = 0.001;

        double &p = q(i, j, k, 0);

        double &pip = q(i + 1, j, k, 0);
        double &pim = q(i - 1, j, k, 0);

        double &pjp = q(i, j + 1, k, 0);
        double &pjm = q(i, j - 1, k, 0);

        double &pkp = q(i, j, k + 1, 0);
        double &pkm = q(i, j, k - 1, 0);

        double ri = abs(pip - 2.0 * p + pim) /
                    ((1.0 - eps) * (abs(pip - p) + abs(p - pim)) +
                     eps * (pip + 2.0 * p + pim));
        phi(i, j, k, 0) = ri;

        double rj = abs(pjp - 2.0 * p + pjm) /
                    ((1.0 - eps) * (abs(pjp - p) + abs(p - pjm)) +
                     eps * (pjp + 2.0 * p + pjm));
        phi(i, j, k, 1) = rj;

        double rk = abs(pkp - 2.0 * p + pkm) /
                    ((1.0 - eps) * (abs(pkp - p) + abs(p - pkm)) +
                     eps * (pkp + 2.0 * p + pkm));
        phi(i, j, k, 2) = rk;
      });
}
