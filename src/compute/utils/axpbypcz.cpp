#include "kernel.hpp"

// The linear combinations of solution registers every time integration stage
// is built from: A = a*A + b*B [+ c*C]. A leading coefficient of zero means
// the stage starts from somewhere else, so A is written without being read;
// the test is on a value that is the same for every element, so the branch
// costs nothing. The interior only: every halo a stage reads is rebuilt by
// the consistify that follows it.

PG_RANGE(cellCenters, components = ne)
struct axpbypcz {
  cellCenterInOut A;
  cellCenterIn B, C;
  double a, b;
  double c;
  KOKKOS_INLINE_FUNCTION void operator()(const int l) const {
    A(l) = a == 0.0 ? b * B(l) + c * C(l) : a * A(l) + b * B(l) + c * C(l);
  }
};

PG_ABI void pgAxpbypcz(const axpbypcz &k, const pgTiling &t) {
  forCellsAndComponents("axpbypcz", t, k);
}
