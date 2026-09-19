#include "kernel.hpp"

// The residual between Q and Q0, the state a dual time stage began from,
// into rMax[ne], rSum[ne]; python combines the ranks.

// the largest residual and the sum of squares, reduced together
struct maxAndSum {
  fpdtype mx, sm;
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
  cellVecIn Q, Q0;
};
// one component's residual; flat, so the shape can pin its columns
struct residualOf {
  cellVecIn Q, Q0;
  int m;
  KOKKOS_INLINE_FUNCTION void operator()(maxAndSum &v) const {
    const fpdtype res = abs(Q(m) - Q0(m));
    v.mx = fmax(res, v.mx);
    v.sm += res * res;
  }
};

PG_ABI void pgResidual(const residual &k, const pgTiling &t, fpdtype *rMax,
                       fpdtype *rSum) {
  for (int m = 0; m < ne; m++) {
    maxAndSum total{0.0, 0.0};
    reduceCells("residual", t, residualOf{k.Q, k.Q0, m},
                residualReducer(total));
    rMax[m] = total.mx, rSum[m] = total.sm;
  }
}
