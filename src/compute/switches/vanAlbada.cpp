#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkos_types.hpp"
#include <math.h>
#include <numeric>

void vanAlbadaEntropy(block_ b) {

  MDRange3 range_cc({b.ng, b.ng, b.ng},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});

  Kokkos::parallel_for(
      "Compute switch from entropy", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double s =
            log(b.q(i, j, k, 0) / pow(b.Q(i, j, k, 0), b.qh(i, j, k, 0)));

        double sip = log(b.q(i + 1, j, k, 0) /
                         pow(b.Q(i + 1, j, k, 0), b.qh(i + 1, j, k, 0)));
        double sim = log(b.q(i - 1, j, k, 0) /
                         pow(b.Q(i - 1, j, k, 0), b.qh(i - 1, j, k, 0)));

        double sjp = log(b.q(i, j + 1, k, 0) /
                         pow(b.Q(i, j + 1, k, 0), b.qh(i, j + 1, k, 0)));
        double sjm = log(b.q(i, j - 1, k, 0) /
                         pow(b.Q(i, j - 1, k, 0), b.qh(i, j - 1, k, 0)));

        double skp = log(b.q(i, j, k + 1, 0) /
                         pow(b.Q(i, j, k + 1, 0), b.qh(i, j, k + 1, 0)));
        double skm = log(b.q(i, j, k - 1, 0) /
                         pow(b.Q(i, j, k - 1, 0), b.qh(i, j, k - 1, 0)));

        double ri = (s - sim + 1e-16) / (sip - s + 1e-16);
        b.phi(i, j, k, 0) = 1.0 - (ri + abs(ri)) / (1.0 + pow(ri, 2.0));

        double rj = (s - sjm + 1e-16) / (sjp - s + 1e-16);
        b.phi(i, j, k, 1) = 1.0 - (rj + abs(rj)) / (1.0 + pow(rj, 2.0));

        double rk = (s - skm + 1e-16) / (skp - s + 1e-16);
        b.phi(i, j, k, 2) = 1.0 - (rk + abs(rk)) / (1.0 + pow(rk, 2.0));
      });
}

void vanAlbadaPressure(block_ b) {

  MDRange3 range_cc({b.ng, b.ng, b.ng},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});

  Kokkos::parallel_for(
      "Compute switch from entropy", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double &p = b.q(i, j, k, 0);

        double &pip = b.q(i + 1, j, k, 0);
        double &pim = b.q(i - 1, j, k, 0);

        double &pjp = b.q(i, j + 1, k, 0);
        double &pjm = b.q(i, j - 1, k, 0);

        double &pkp = b.q(i, j, k + 1, 0);
        double &pkm = b.q(i, j, k - 1, 0);

        double ri = (p - pim + 1e-16) / (pip - p + 1e-16);
        b.phi(i, j, k, 0) = 1.0 - (ri + abs(ri)) / (1.0 + pow(ri, 2.0));

        double rj = (p - pjm + 1e-16) / (pjp - p + 1e-16);
        b.phi(i, j, k, 1) = 1.0 - (rj + abs(rj)) / (1.0 + pow(rj, 2.0));

        double rk = (p - pkm + 1e-16) / (pkp - p + 1e-16);
        b.phi(i, j, k, 2) = 1.0 - (rk + abs(rk)) / (1.0 + pow(rk, 2.0));
      });
}
