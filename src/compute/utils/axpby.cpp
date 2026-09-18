#include "kernel.hpp"

// A = a*A + b*dt*B over every element of the allocations, as copy: a
// Runge-Kutta stage's step by the derivative, with the step's dt read where
// the kernels run so the weights are a stage's constants. No case for a zero
// coefficient: an array is zeroed when it is made and holds a state after,
// so 0 * A is 0, and a branch on the coefficient cost a fifth of the time.
// The arrays are ne wide.
PG_RANGE(elements, components = ne)
struct axpby {
  cellCenterInOut A;
  fpdtype a, b;
  cellCenterIn B;
  caseIn dt;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    A[i] = a * A[i] + b * dt() * B[i];
  }
};

PG_ABI void pgAxpby(const axpby &k, const pgTiling &t) {
  forElements("axpby", t, k);
}
