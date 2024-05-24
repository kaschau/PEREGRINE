#include "block_.hpp"
#include "kokkosTypes.hpp"
#include "math.h"
#include "thtrdat_.hpp"
#include <Kokkos_Core.hpp>

static void computeFlux(const block_ &b, fourDview &iF, const threeDview &iS,
                        const threeDview &isx, const threeDview &isy,
                        const threeDview &isz, const threeDview &inx,
                        const threeDview &iny, const threeDview &inz,
                        const int iMod, const int jMod, const int kMod) {

  const double kappa2 = 0.5;
  const double kappa4 = 0.005;

  // face flux range
  MDRange3 range(
      {b.ng, b.ng, b.ng},
      {b.ni + b.ng - 1 + iMod, b.nj + b.ng - 1 + jMod, b.nk + b.ng - 1 + kMod});

  Kokkos::parallel_for(
      "Scalar Dissipation face conv fluxes", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // The weird mod indexing math is so we grab the correct last
        // phi index for each dimension
        const int phiIndex =
            (iMod - 1) * iMod + (jMod)*jMod + (kMod + 1) * kMod;
        const double eps2 =
            kappa2 * fmax(b.phi(i, j, k, phiIndex),
                          b.phi(i - iMod, j - jMod, k - kMod, phiIndex));
        const double eps4 = fmax(0.0, kappa4 - eps2);

        // Compute face normal volume flux vector
        const double uf =
            0.5 * (b.q(i, j, k, 1) + b.q(i - iMod, j - jMod, k - kMod, 1));
        const double vf =
            0.5 * (b.q(i, j, k, 2) + b.q(i - iMod, j - jMod, k - kMod, 2));
        const double wf =
            0.5 * (b.q(i, j, k, 3) + b.q(i - iMod, j - jMod, k - kMod, 3));

        const double U =
            inx(i, j, k) * uf + iny(i, j, k) * vf + inz(i, j, k) * wf;

        const double a =
            (abs(U) +
             0.5 * (b.qh(i, j, k, 3) + b.qh(i - iMod, j - jMod, k - kMod, 3))) *
            iS(i, j, k);

        double rho2, rho4;
        rho2 = b.Q(i, j, k, 0) - b.Q(i - iMod, j - jMod, k - kMod, 0);
        rho4 = b.Q(i + 1, j, k, 0) - 3.0 * b.Q(i, j, k, 0) +
               3.0 * b.Q(i - iMod, j - jMod, k - kMod, 0) -
               b.Q(i - iMod * 2, j - jMod * 2, k - kMod * 2, 0);

        // Continuity dissipation
        iF(i, j, k, 0) = a * (eps2 * rho2 - eps4 * rho4);

        // u momentum dissipation
        double u2, u4;
        u2 = b.Q(i, j, k, 1) - b.Q(i - iMod, j - jMod, k - kMod, 1);
        u4 = b.Q(i + 1, j, k, 1) - 3.0 * b.Q(i, j, k, 1) +
             3.0 * b.Q(i - iMod, j - jMod, k - kMod, 1) -
             b.Q(i - iMod * 2, j - jMod * 2, k - kMod * 2, 1);

        iF(i, j, k, 1) = a * (eps2 * u2 - eps4 * u4);

        // v momentum dissipation
        double v2, v4;
        v2 = b.Q(i, j, k, 2) - b.Q(i - iMod, j - jMod, k - kMod, 2);
        v4 = b.Q(i + 1, j, k, 2) - 3.0 * b.Q(i, j, k, 2) +
             3.0 * b.Q(i - iMod, j - jMod, k - kMod, 2) -
             b.Q(i - iMod * 2, j - jMod * 2, k - kMod * 2, 2);

        iF(i, j, k, 2) = a * (eps2 * v2 - eps4 * v4);

        // w momentum dissipation
        double w2, w4;
        w2 = b.Q(i, j, k, 3) - b.Q(i - iMod, j - jMod, k - kMod, 3);
        w4 = b.Q(i + 1, j, k, 3) - 3.0 * b.Q(i, j, k, 3) +
             3.0 * b.Q(i - iMod, j - jMod, k - kMod, 3) -
             b.Q(i - iMod * 2, j - jMod * 2, k - kMod * 2, 3);

        iF(i, j, k, 3) = a * (eps2 * w2 - eps4 * w4);

        // total energy dissipation
        double e2, e4;
        e2 = b.Q(i, j, k, 4) - b.Q(i - iMod, j - jMod, k - kMod, 4);
        e4 = b.Q(i + 1, j, k, 4) - 3.0 * b.Q(i, j, k, 4) +
             3.0 * b.Q(i - iMod, j - jMod, k - kMod, 4) -
             b.Q(i - iMod * 2, j - jMod * 2, k - kMod * 2, 4);

        iF(i, j, k, 4) = a * (eps2 * e2 - eps4 * e4);

        // Species
        for (int n = 0; n < b.ne - 5; n++) {
          double Y2, Y4;
          Y2 = b.Q(i, j, k, 5 + n) - b.Q(i - iMod, j - jMod, k - kMod, 5 + n);
          Y4 = b.Q(i + 1, j, k, 5 + n) - 3.0 * b.Q(i, j, k, 5 + n) +
               3.0 * b.Q(i - iMod, j - jMod, k - kMod, 5 + n) -
               b.Q(i - iMod * 2, j - jMod * 2, k - kMod * 2, 5 + n);
          iF(i, j, k, 5 + n) = a * (eps2 * Y2 - eps4 * Y4);
        }
      });
}

void scalarDissipation(block_ &b) {
  computeFlux(b, b.iF, b.iS, b.isx, b.isy, b.isz, b.inx, b.iny, b.inz, 1, 0, 0);
  computeFlux(b, b.jF, b.jS, b.jsx, b.jsy, b.jsz, b.jnx, b.jny, b.jnz, 0, 1, 0);
  computeFlux(b, b.kF, b.kS, b.ksx, b.ksy, b.ksz, b.knx, b.kny, b.knz, 0, 0, 1);
}
