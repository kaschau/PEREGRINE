#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <math.h>
#include <numeric>

PG_ABI void pgVanAlbadaPressure(const pgView *phi_, const pgView *q_,
                                const pgDims *d) {
  auto phi = as4(*phi_);
  auto q = as4(*q_);
  const int ni = d->ni, nj = d->nj, nk = d->nk;

  MDRange3 range_cc({ng, ng, ng}, {ni + ng - 1, nj + ng - 1, nk + ng - 1});

  Kokkos::parallel_for(
      "Compute switch from entropy", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double &p = q(i, j, k, 0);

        double &pip = q(i + 1, j, k, 0);
        double &pim = q(i - 1, j, k, 0);

        double &pjp = q(i, j + 1, k, 0);
        double &pjm = q(i, j - 1, k, 0);

        double &pkp = q(i, j, k + 1, 0);
        double &pkm = q(i, j, k - 1, 0);

        double ri = (p - pim + 1e-16) / (pip - p + 1e-16);
        phi(i, j, k, 0) = 1.0 - (ri + abs(ri)) / (1.0 + pow(ri, 2.0));

        double rj = (p - pjm + 1e-16) / (pjp - p + 1e-16);
        phi(i, j, k, 1) = 1.0 - (rj + abs(rj)) / (1.0 + pow(rj, 2.0));

        double rk = (p - pkm + 1e-16) / (pkp - p + 1e-16);
        phi(i, j, k, 2) = 1.0 - (rk + abs(rk)) / (1.0 + pow(rk, 2.0));
      });
}
