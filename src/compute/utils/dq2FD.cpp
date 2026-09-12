#include "abi.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

PG_ABI void pgDq2FD(const pgView *dENCdxyz_, const pgView *grads_,
                    const pgView *q_, const pgDims *d) {
  auto dENCdxyz = as5(*dENCdxyz_);
  auto grads = as5(*grads_);
  auto q = as4(*q_);
  const int ng = d->ng, ni = d->ni, nj = d->nj, nk = d->nk;
  const int ne = q.extent(3);

  //-------------------------------------------------------------------------------------------|
  // Spatial derivatices of primative variables
  // estimated via second order finite difference
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({ng, ng, ng, 0},
                    {ni + ng - 1, nj + ng - 1, nk + ng - 1, ne});
  Kokkos::parallel_for(
      "2nd order spatial deriv", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        double dqdENC[3] = {0.5 * (q(i + 1, j, k, l) - q(i - 1, j, k, l)),
                            0.5 * (q(i, j + 1, k, l) - q(i, j - 1, k, l)),
                            0.5 * (q(i, j, k + 1, l) - q(i, j, k - 1, l))};

        for (int d = 0; d < 3; d++) {
          double grad = 0.0;
          for (int e = 0; e < 3; e++) {
            grad += dqdENC[e] * dENCdxyz(i, j, k, e, d);
          }
          grads(i, j, k, l, d) = grad;
        }
      });
}
