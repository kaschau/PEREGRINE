#include "kernel.hpp"

PG_RANGE(interior, ne)
struct dq2FD {
  in dENCdxyz, q;
  out grads;
  dims d;
  KOKKOS_INLINE_FUNCTION void operator()(const int l) const {
    const int ni = d->ni, nj = d->nj, nk = d->nk;
    //-------------------------------------------------------------------------------------------|
    // Spatial derivatices of primative variables
    // estimated via second order finite difference
    //-------------------------------------------------------------------------------------------|

    double dqdENC[3] = {0.5 * (q(+I, l) - q(-I, l)),
                        0.5 * (q(+J, l) - q(-J, l)),
                        0.5 * (q(+K, l) - q(-K, l))};

    for (int d = 0; d < 3; d++) {
      double grad = 0.0;
      for (int e = 0; e < 3; e++) {
        grad += dqdENC[e] * dENCdxyz(e, d);
      }
      grads(l, d) = grad;
    }
  }
};

PG_ABI void pgDq2FD(const dq2FD &k, const pgTiling &t) {
  forCellsAndComponents("2nd order spatial deriv", t, k);
}
