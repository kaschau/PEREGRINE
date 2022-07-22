#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkos_types.hpp"
#include "math.h"
#include "thtrdat_.hpp"

void scalarDissipation(block_ b) {

  const double kappa2 = 0.5;
  const double kappa4 = 0.005;

  //-------------------------------------------------------------------------------------------|
  // i flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_i({b.ng, b.ng, b.ng},
                   {b.ni + b.ng, b.nj + b.ng - 1, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "Scalar Dissipation i face conv fluxes", range_i,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        const double eps2 =
            kappa2 * fmax(b.phi(i, j, k, 0), b.phi(i - 1, j, k, 0));
        const double eps4 = fmax(0.0, kappa4 - eps2);

        // Compute face normal volume flux vector
        const double uf = 0.5 * (b.q(i, j, k, 1) + b.q(i - 1, j, k, 1));
        const double vf = 0.5 * (b.q(i, j, k, 2) + b.q(i - 1, j, k, 2));
        const double wf = 0.5 * (b.q(i, j, k, 3) + b.q(i - 1, j, k, 3));

        const double U =
            b.inx(i, j, k) * uf + b.iny(i, j, k) * vf + b.inz(i, j, k) * wf;

        const double a =
            (abs(U) + 0.5 * (b.qh(i, j, k, 3) + b.qh(i - 1, j, k, 3))) *
            b.iS(i, j, k);

        double rho2, rho4;
        rho2 = b.Q(i, j, k, 0) - b.Q(i - 1, j, k, 0);
        rho4 = b.Q(i + 1, j, k, 0) - 3.0 * b.Q(i, j, k, 0) +
               3.0 * b.Q(i - 1, j, k, 0) - b.Q(i - 2, j, k, 0);

        // Continuity dissipation
        b.iF(i, j, k, 0) = a * (eps2 * rho2 - eps4 * rho4);

        // u momentum dissipation
        double u2, u4;
        u2 = b.Q(i, j, k, 1) - b.Q(i - 1, j, k, 1);
        u4 = b.Q(i + 1, j, k, 1) - 3.0 * b.Q(i, j, k, 1) +
             3.0 * b.Q(i - 1, j, k, 1) - b.Q(i - 2, j, k, 1);

        b.iF(i, j, k, 1) = a * (eps2 * u2 - eps4 * u4);

        // v momentum dissipation
        double v2, v4;
        v2 = b.Q(i, j, k, 2) - b.Q(i - 1, j, k, 2);
        v4 = b.Q(i + 1, j, k, 2) - 3.0 * b.Q(i, j, k, 2) +
             3.0 * b.Q(i - 1, j, k, 2) - b.Q(i - 2, j, k, 2);

        b.iF(i, j, k, 2) = a * (eps2 * v2 - eps4 * v4);

        // w momentum dissipation
        double w2, w4;
        w2 = b.Q(i, j, k, 3) - b.Q(i - 1, j, k, 3);
        w4 = b.Q(i + 1, j, k, 3) - 3.0 * b.Q(i, j, k, 3) +
             3.0 * b.Q(i - 1, j, k, 3) - b.Q(i - 2, j, k, 3);

        b.iF(i, j, k, 3) = a * (eps2 * w2 - eps4 * w4);

        // total energy dissipation
        double e2, e4;
        e2 = b.Q(i, j, k, 4) - b.Q(i - 1, j, k, 4);
        e4 = b.Q(i + 1, j, k, 4) - 3.0 * b.Q(i, j, k, 4) +
             3.0 * b.Q(i - 1, j, k, 4) - b.Q(i - 2, j, k, 4);

        b.iF(i, j, k, 4) = a * (eps2 * e2 - eps4 * e4);

        // Species
        for (int n = 0; n < b.ne - 5; n++) {
          double Y2, Y4;
          Y2 = b.Q(i, j, k, 5 + n) - b.Q(i - 1, j, k, 5 + n);
          Y4 = b.Q(i + 1, j, k, 5 + n) - 3.0 * b.Q(i, j, k, 5 + n) +
               3.0 * b.Q(i - 1, j, k, 5 + n) - b.Q(i - 2, j, k, 5 + n);
          b.iF(i, j, k, 5 + n) = a * (eps2 * Y2 - eps4 * Y4);
        }
      });

  //-------------------------------------------------------------------------------------------|
  // j flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_j({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "Scalar Dissipation j face conv fluxes", range_j,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        const double eps2 =
            kappa2 * fmax(b.phi(i, j, k, 1), b.phi(i, j - 1, k, 1));
        const double eps4 = fmax(0.0, kappa4 - eps2);

        // Compute face normal volume flux vector
        const double uf = 0.5 * (b.q(i, j, k, 1) + b.q(i, j - 1, k, 1));
        const double vf = 0.5 * (b.q(i, j, k, 2) + b.q(i, j - 1, k, 2));
        const double wf = 0.5 * (b.q(i, j, k, 3) + b.q(i, j - 1, k, 3));

        const double U =
            b.jnx(i, j, k) * uf + b.jny(i, j, k) * vf + b.jnz(i, j, k) * wf;

        const double a =
            (abs(U) + 0.5 * (b.qh(i, j, k, 3) + b.qh(i, j - 1, k, 3))) *
            b.jS(i, j, k);

        double rho2, rho4;
        rho2 = b.Q(i, j, k, 0) - b.Q(i, j - 1, k, 0);
        rho4 = b.Q(i, j + 1, k, 0) - 3.0 * b.Q(i, j, k, 0) +
               3.0 * b.Q(i, j - 1, k, 0) - b.Q(i, j - 2, k, 0);

        // Continuity dissipation
        b.jF(i, j, k, 0) = a * (eps2 * rho2 - eps4 * rho4);

        // u momentum dissipation
        double u2, u4;
        u2 = b.Q(i, j, k, 1) - b.Q(i, j - 1, k, 1);
        u4 = b.Q(i, j + 1, k, 1) - 3.0 * b.Q(i, j, k, 1) +
             3.0 * b.Q(i, j - 1, k, 1) - b.Q(i, j - 2, k, 1);

        b.jF(i, j, k, 1) = a * (eps2 * u2 - eps4 * u4);

        // v momentum dissipation
        double v2, v4;
        v2 = b.Q(i, j, k, 2) - b.Q(i, j - 1, k, 2);
        v4 = b.Q(i, j + 1, k, 2) - 3.0 * b.Q(i, j, k, 2) +
             3.0 * b.Q(i, j - 1, k, 2) - b.Q(i, j - 2, k, 2);

        b.jF(i, j, k, 2) = a * (eps2 * v2 - eps4 * v4);

        // w momentum dissipation
        double w2, w4;
        w2 = b.Q(i, j, k, 3) - b.Q(i, j - 1, k, 3);
        w4 = b.Q(i, j + 1, k, 3) - 3.0 * b.Q(i, j, k, 3) +
             3.0 * b.Q(i, j - 1, k, 3) - b.Q(i, j - 2, k, 3);

        b.jF(i, j, k, 3) = a * (eps2 * w2 - eps4 * w4);

        // total energy dissipation
        double e2, e4;
        e2 = b.Q(i, j, k, 4) - b.Q(i, j - 1, k, 4);
        e4 = b.Q(i, j + 1, k, 4) - 3.0 * b.Q(i, j, k, 4) +
             3.0 * b.Q(i, j - 1, k, 4) - b.Q(i, j - 2, k, 4);

        b.jF(i, j, k, 4) = a * (eps2 * e2 - eps4 * e4);

        // Species
        for (int n = 0; n < b.ne - 5; n++) {
          double Y2, Y4;
          Y2 = b.Q(i, j, k, 5 + n) - b.Q(i, j - 1, k, 5 + n);
          Y4 = b.Q(i, j + 1, k, 5 + n) - 3.0 * b.Q(i, j, k, 5 + n) +
               3.0 * b.Q(i, j - 1, k, 5 + n) - b.Q(i, j - 2, k, 5 + n);
          b.jF(i, j, k, 5 + n) = a * (eps2 * Y2 - eps4 * Y4);
        }
      });

  //-------------------------------------------------------------------------------------------|
  // k flux face range
  //-------------------------------------------------------------------------------------------|
  MDRange3 range_k({b.ng, b.ng, b.ng},
                   {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng});
  Kokkos::parallel_for(
      "Scalar Dissipation k face conv fluxes", range_k,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        const double eps2 =
            kappa2 * fmax(b.phi(i, j, k, 2), b.phi(i, j, k - 1, 2));
        const double eps4 = fmax(0.0, kappa4 - eps2);

        // Compute face normal volume flux vector
        const double uf = 0.5 * (b.q(i, j, k, 1) + b.q(i, j, k - 1, 1));
        const double vf = 0.5 * (b.q(i, j, k, 2) + b.q(i, j, k - 1, 2));
        const double wf = 0.5 * (b.q(i, j, k, 3) + b.q(i, j, k - 1, 3));

        const double U =
            b.knx(i, j, k) * uf + b.kny(i, j, k) * vf + b.knz(i, j, k) * wf;

        const double a =
            (abs(U) + 0.5 * (b.qh(i, j, k, 3) + b.qh(i, j, k - 1, 3))) *
            b.kS(i, j, k);

        double rho2, rho4;
        rho2 = b.Q(i, j, k, 0) - b.Q(i, j, k - 1, 0);
        rho4 = b.Q(i, j, k + 1, 0) - 3.0 * b.Q(i, j, k, 0) +
               3.0 * b.Q(i, j, k - 1, 0) - b.Q(i, j, k - 2, 0);

        // Continuity dissipation
        b.kF(i, j, k, 0) = a * (eps2 * rho2 - eps4 * rho4);

        // u momentum dissipation
        double u2, u4;
        u2 = b.Q(i, j, k, 1) - b.Q(i, j, k - 1, 1);
        u4 = b.Q(i, j, k + 1, 1) - 3.0 * b.Q(i, j, k, 1) +
             3.0 * b.Q(i, j, k - 1, 1) - b.Q(i, j, k - 2, 1);

        b.kF(i, j, k, 1) = a * (eps2 * u2 - eps4 * rho4 * u4);

        // v momentum dissipation
        double v2, v4;
        v2 = b.Q(i, j, k, 2) - b.Q(i, j, k - 1, 2);
        v4 = b.Q(i, j, k + 1, 2) - 3.0 * b.Q(i, j, k, 2) +
             3.0 * b.Q(i, j, k - 1, 2) - b.Q(i, j, k - 2, 2);

        b.kF(i, j, k, 2) = a * (eps2 * v2 - eps4 * v4);

        // w momentum dissipation
        double w2, w4;
        w2 = b.Q(i, j, k, 3) - b.Q(i, j, k - 1, 3);
        w4 = b.Q(i, j, k + 1, 3) - 3.0 * b.Q(i, j, k, 3) +
             3.0 * b.Q(i, j, k - 1, 3) - b.Q(i, j, k - 2, 3);

        b.kF(i, j, k, 3) = a * (eps2 * w2 - eps4 * w4);

        // total energy dissipation
        double e2, e4;
        e2 = b.Q(i, j, k, 4) - b.Q(i, j, k - 1, 4);
        e4 = b.Q(i, j, k + 1, 4) - 3.0 * b.Q(i, j, k, 4) +
             3.0 * b.Q(i, j, k - 1, 4) - b.Q(i, j, k - 2, 4);

        b.kF(i, j, k, 4) = a * (eps2 * e2 - eps4 * e4);

        // Species
        for (int n = 0; n < b.ne - 5; n++) {
          double Y2, Y4;
          Y2 = b.Q(i, j, k, 5 + n) - b.Q(i, j, k - 1, 5 + n);
          Y4 = b.Q(i, j, k + 1, 5 + n) - 3.0 * b.Q(i, j, k, 5 + n) +
               3.0 * b.Q(i, j, k - 1, 5 + n) - b.Q(i, j, k - 2, 5 + n);
          b.kF(i, j, k, 5 + n) = a * (eps2 * Y2 - eps4 * Y4);
        }
      });
}
