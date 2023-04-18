#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include "thtrdat_.hpp"

void KEEPdissipation(block_ &b) {

  const double kappa2 = 0.5;

  //-------------------------------------------------------------------------------------------|
  // i flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_i({b.ng, b.ng, b.ng},
                   {b.ni + b.ng, b.nj + b.ng - 1, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "Scalar Dissipation i face conv fluxes", range_i,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        const double eps2 =
            fmin(1.0, kappa2 * fmax(b.phi(i, j, k, 0), b.phi(i - 1, j, k, 0)));

        // Compute face normal volume flux vector
        const double uf = 0.5 * (b.q(i, j, k, 1) + b.q(i - 1, j, k, 1));
        const double vf = 0.5 * (b.q(i, j, k, 2) + b.q(i - 1, j, k, 2));
        const double wf = 0.5 * (b.q(i, j, k, 3) + b.q(i - 1, j, k, 3));

        const double U =
            b.inx(i, j, k) * uf + b.iny(i, j, k) * vf + b.inz(i, j, k) * wf;

        const double a =
            (abs(U) + 0.5 * (b.qh(i, j, k, 3) + b.qh(i - 1, j, k, 3))) *
            b.iS(i, j, k);

        double rho2;
        rho2 = b.Q(i, j, k, 0) - b.Q(i - 1, j, k, 0);

        // Continuity dissipation
        double Dc = (a * eps2) * rho2;
        b.iF(i, j, k, 0) = Dc;

        // u momentum dissipation
        b.iF(i, j, k, 1) = Dc * 0.5 * (b.q(i, j, k, 1) + b.q(i - 1, j, k, 1));

        // v momentum dissipation
        b.iF(i, j, k, 2) = Dc * 0.5 * (b.q(i, j, k, 2) + b.q(i - 1, j, k, 2));

        // w momentum dissipation
        b.iF(i, j, k, 3) = Dc * 0.5 * (b.q(i, j, k, 3) + b.q(i - 1, j, k, 3));

        // kinetic energy dissipation
        b.iF(i, j, k, 4) = Dc * 0.5 *
                           (b.q(i, j, k, 1) * b.q(i - 1, j, k, 1) +
                            b.q(i, j, k, 2) * b.q(i - 1, j, k, 2) +
                            b.q(i, j, k, 3) * b.q(i - 1, j, k, 3));

        // TODO internal energy dissipation
        double e2 = b.qh(i, j, k, 4) - b.qh(i - 1, j, k, 4);
        b.iF(i, j, k, 4) += a * eps2 * e2;

        // TODO Species???
        for (int n = 0; n < b.ne - 5; n++) {
          b.iF(i, j, k, 5 + n) =
              Dc * 0.5 * b.Q(i, j, k, 5 + n) + b.Q(i - 1, j, k, 5 + n);
        }
      });
}
