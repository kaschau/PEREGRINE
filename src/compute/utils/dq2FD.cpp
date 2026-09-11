#include "block_.hpp"
#include "compute.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

void dq2FD(block_ &b) {

  //-------------------------------------------------------------------------------------------|
  // Spatial derivatices of primative variables
  // estimated via second order finite difference
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "2nd order spatial deriv", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        double dqdENC[3] = {0.5 * (b.q(i + 1, j, k, l) - b.q(i - 1, j, k, l)),
                            0.5 * (b.q(i, j + 1, k, l) - b.q(i, j - 1, k, l)),
                            0.5 * (b.q(i, j, k + 1, l) - b.q(i, j, k - 1, l))};

        for (int d = 0; d < 3; d++) {
          double grad = 0.0;
          for (int e = 0; e < 3; e++) {
            grad += dqdENC[e] * b.dENCdxyz(i, j, k, e, d);
          }
          b.grads(i, j, k, l, d) = grad;
        }
      });
}
