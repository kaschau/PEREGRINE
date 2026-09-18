#include "kernel.hpp"

// A = a*A + b*B + c*dt*C, the three-array combination a Runge-Kutta stage
// is -- C the derivative, stepped by the step's dt read where the kernels
// run -- over every element of the allocations, as copy; no case for a zero
// coefficient, as in axpby. The arrays are ne wide.
PG_RANGE(elements, components = ne)
struct axpbypcz {
  cellCenterInOut A;
  cellCenterIn B, C;
  fpdtype a, b;
  fpdtype c;
  caseIn dt;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    A[i] = a * A[i] + b * B[i] + c * dt() * C[i];
  }
};

PG_ABI void pgAxpbypcz(const axpbypcz &k, const pgTiling &t) {
  forElements("axpbypcz", t, k);
}
