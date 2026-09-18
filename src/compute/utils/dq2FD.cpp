#include "kernel.hpp"

// The gradient of each primitive a diffusive flux needs, by second order
// central differences: slots 0 .. 2 the velocity, 3 the temperature, 4 + n
// the mass fractions Y(n) of the first ns - 1 species. A thread does a
// cell and walks the slots: the neighbours' densities, their one reciprocal
// per direction and the metrics are the cell's, not each slot's. The
// velocity and Y are read off Q at each stencil point; a difference of two
// quotients is one division.
PG_RANGE(cellCenters)
struct dq2FD {
  cellCenterIn dENCdxyz, Q, q;
  cellCenterOut grads;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    // each direction's two neighbour densities and 1 / their product
    const fpdtype rIp = Q(+I, 0), rIm = Q(-I, 0), invI = 1.0 / (rIp * rIm);
    const fpdtype rJp = Q(+J, 0), rJm = Q(-J, 0), invJ = 1.0 / (rJp * rJm);
    const fpdtype rKp = Q(+K, 0), rKm = Q(-K, 0), invK = 1.0 / (rKp * rKm);
    fpdtype metric[3][3];
    for (int e = 0; e < 3; e++) {
      for (int d = 0; d < 3; d++) {
        metric[e][d] = dENCdxyz(e, d);
      }
    }
    // the difference in each computational direction, then the physical
    // gradient through the metrics
    const auto gradient = [&](const int l, const fpdtype *dqdENC) {
      for (int d = 0; d < 3; d++) {
        grads(l, d) = dqdENC[0] * metric[0][d] + dqdENC[1] * metric[1][d] +
                      dqdENC[2] * metric[2][d];
      }
    };
    for (int l = 0; l < ne - 1; l++) {
      fpdtype dqdENC[3];
      if (l == 3) {
        dqdENC[0] = 0.5 * (q(+I, 1) - q(-I, 1));
        dqdENC[1] = 0.5 * (q(+J, 1) - q(-J, 1));
        dqdENC[2] = 0.5 * (q(+K, 1) - q(-K, 1));
      } else {
        // the conserved slot of the same quantity, rho times it: the
        // momentum for a velocity, the species mass for a Y; both sit one
        // past the slot
        const int m = l + 1;
        dqdENC[0] = 0.5 * (Q(+I, m) * rIm - Q(-I, m) * rIp) * invI;
        dqdENC[1] = 0.5 * (Q(+J, m) * rJm - Q(-J, m) * rJp) * invJ;
        dqdENC[2] = 0.5 * (Q(+K, m) * rKm - Q(-K, m) * rKp) * invK;
      }
      gradient(l, dqdENC);
    }
  }
};

PG_ABI void pgDq2FD(const dq2FD &k, const pgTiling &t) {
  forCells("2nd order spatial deriv", t, k);
}
