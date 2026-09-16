#include "kernel.hpp"

// The gradient of each primitive a diffusive flux needs, by second order
// central differences: slots 0 .. 2 the velocity, 3 the temperature, 4 + n
// the mass fractions Y(n) of the first ns - 1 species. The velocity and Y
// are read off Q at each stencil point; a difference of two quotients is
// one division.
PG_RANGE(cellCenters, components = ne - 1)
struct dq2FD {
  cellCenterIn dENCdxyz, Q, q;
  cellCenterOut grads;
  KOKKOS_INLINE_FUNCTION void operator()(const int l) const {
    double dqdENC[3];
    if (l == 3) {
      dqdENC[0] = 0.5 * (q(+I, 1) - q(-I, 1));
      dqdENC[1] = 0.5 * (q(+J, 1) - q(-J, 1));
      dqdENC[2] = 0.5 * (q(+K, 1) - q(-K, 1));
    } else {
      // the conserved slot of the same quantity, rho times it: the momentum
      // for a velocity, the species mass for a Y; both sit one past the slot
      const int m = l + 1;
      dqdENC[0] = 0.5 * (Q(+I, m) * Q(-I, 0) - Q(-I, m) * Q(+I, 0)) /
                  (Q(+I, 0) * Q(-I, 0));
      dqdENC[1] = 0.5 * (Q(+J, m) * Q(-J, 0) - Q(-J, m) * Q(+J, 0)) /
                  (Q(+J, 0) * Q(-J, 0));
      dqdENC[2] = 0.5 * (Q(+K, m) * Q(-K, 0) - Q(-K, m) * Q(+K, 0)) /
                  (Q(+K, 0) * Q(-K, 0));
    }

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
