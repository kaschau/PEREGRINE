#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <math.h>
#include <numeric>

PG_ABI void pgVanAlbadaPressure(int count, pgOut *phi_, pgIn *q_,
                                const pgDims *d) {
  for (int e = 0; e < count; e++) {
    auto phi = as4(phi_[e]);
    auto q = as4(q_[e]);
    const int ni = d[e].ni, nj = d[e].nj, nk = d[e].nk;

    MDRange3 range_cc({ng, ng, ng}, {ni + ng - 1, nj + ng - 1, nk + ng - 1});

    Kokkos::parallel_for(
        "Compute switch from entropy", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          const double &p = q(i, j, k, 0);

          const double &pip = q(i + 1, j, k, 0);
          const double &pim = q(i - 1, j, k, 0);

          const double &pjp = q(i, j + 1, k, 0);
          const double &pjm = q(i, j - 1, k, 0);

          const double &pkp = q(i, j, k + 1, 0);
          const double &pkm = q(i, j, k - 1, 0);

          double ri = (p - pim + 1e-16) / (pip - p + 1e-16);
          phi(i, j, k, 0) = 1.0 - (ri + abs(ri)) / (1.0 + pow(ri, 2.0));

          double rj = (p - pjm + 1e-16) / (pjp - p + 1e-16);
          phi(i, j, k, 1) = 1.0 - (rj + abs(rj)) / (1.0 + pow(rj, 2.0));

          double rk = (p - pkm + 1e-16) / (pkp - p + 1e-16);
          phi(i, j, k, 2) = 1.0 - (rk + abs(rk)) / (1.0 + pow(rk, 2.0));
        });
  }
}
