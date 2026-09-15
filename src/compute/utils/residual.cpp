#include "kernel.hpp"

// The residual between q and Q0 (which holds primitives under dual time),
// into rMax[ne], rSum[ne]; python combines the ranks.

// the largest residual and the sum of squares, reduced together
struct maxAndSum {
  double mx, sm;
};
struct residualReducer {
  using reducer = residualReducer;
  using value_type = maxAndSum;
  using result_view_type =
      Kokkos::View<maxAndSum *, hostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  maxAndSum &value;
  KOKKOS_INLINE_FUNCTION residualReducer(maxAndSum &v) : value(v) {}
  KOKKOS_INLINE_FUNCTION void join(maxAndSum &d, const maxAndSum &s) const {
    d.mx = fmax(d.mx, s.mx), d.sm += s.sm;
  }
  KOKKOS_INLINE_FUNCTION void init(maxAndSum &v) const { v = {0.0, 0.0}; }
  KOKKOS_INLINE_FUNCTION maxAndSum &reference() const { return value; }
  KOKKOS_INLINE_FUNCTION result_view_type view() const {
    return result_view_type(&value, 1);
  }
  KOKKOS_INLINE_FUNCTION bool references_scalar() const { return true; }
};

PG_RANGE(cellCenters)
struct residual {
  cellCenterIn q, Q0;
};
// one component's residual; flat, so the shape can pin its columns
struct residualOf {
  cellCenterIn q, Q0;
  int m;
  KOKKOS_INLINE_FUNCTION void operator()(maxAndSum &v) const {
    const double res = abs(q(m) - Q0(m));
    v.mx = fmax(res, v.mx);
    v.sm += res * res;
  }
};

PG_ABI void pgResidual(const residual &k, const pgTiling &t, double *rMax,
                       double *rSum) {
  for (int m = 0; m < ne; m++) {
    maxAndSum total{0.0, 0.0};
    reduceCells("residual", t, residualOf{k.q, k.Q0, m},
                residualReducer(total));
    rMax[m] = total.mx, rSum[m] = total.sm;
  }
}
