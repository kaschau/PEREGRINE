#include "kernel.hpp"

// One block array into another, A = B, every element of the allocation as
// one flat run: halos included, since every halo a stage reads is rebuilt by
// the consistify that follows it, and that is what moves the bytes at the
// card's copy rate. The arrays are ne wide.
PG_RANGE(elements, components = ne)
struct copy {
  cellCenterOut A;
  cellCenterIn B;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const { A[i] = B[i]; }
};

PG_ABI void pgCopy(const copy &k, const pgTiling &t) {
  forElements("copy", t, k);
}
