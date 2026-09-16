#include "kernel.hpp"

// A = a*A + b*B + c*C, the three-array combination a Runge-Kutta stage is,
// over every element of the allocations, as copy; no case for a zero
// coefficient, as in axpby. The arrays are ne wide.
PG_RANGE(elements, components = ne)
struct axpbypcz {
  cellCenterInOut A;
  cellCenterIn B, C;
  double a, b;
  double c;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    A[i] = a * A[i] + b * B[i] + c * C[i];
  }
};

PG_ABI void pgAxpbypcz(const axpbypcz &k, const pgTiling &t) {
  forElements("axpbypcz", t, k);
}
