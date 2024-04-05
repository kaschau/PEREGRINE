#include "block_.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>
#include <math.h>
#include <numeric>

void jamesonPressure(block_ &b) {

  MDRange3 range_cc({b.ng, b.ng, b.ng},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});

  Kokkos::parallel_for(
      "Compute switch from pressure", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double &p = b.q(i, j, k, 0);

        double &pip = b.q(i + 1, j, k, 0);
        double &pim = b.q(i - 1, j, k, 0);

        double &pjp = b.q(i, j + 1, k, 0);
        double &pjm = b.q(i, j - 1, k, 0);

        double &pkp = b.q(i, j, k + 1, 0);
        double &pkm = b.q(i, j, k - 1, 0);

        double ri = abs(pip - 2.0 * p + pim) / abs(pip + 2.0 * p + pim);
        b.phi(i, j, k, 0) = ri;

        double rj = abs(pjp - 2.0 * p + pjm) / abs(pjp + 2.0 * p + pjm);
        b.phi(i, j, k, 1) = rj;

        double rk = abs(pkp - 2.0 * p + pkm) / abs(pkp + 2.0 * p + pkm);
        b.phi(i, j, k, 2) = rk;
      });
}
