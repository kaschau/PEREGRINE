#include "kernel.hpp"

// A = a*A + b*B over every element of the allocations, as copy. No case for
// a zero coefficient: an array is zeroed when it is made and holds a state
// after, so 0 * A is 0, and a branch on the coefficient cost a fifth of the
// time. The arrays are ne wide.
PG_RANGE(elements, components = ne)
struct axpby {
  cellCenterInOut A;
  double a, b;
  cellCenterIn B;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    A[i] = a * A[i] + b * B[i];
  }
};

PG_ABI void pgAxpby(const axpby &k, const pgTiling &t) {
  forElements("axpby", t, k);
}
