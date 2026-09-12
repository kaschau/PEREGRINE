#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

// The linear combinations of solution registers every time integration stage
// is built from: A = a*A + b*B [+ c*C]. A leading coefficient of zero means
// the stage starts from somewhere else, so A is written without being read;
// the test is on a value that is the same for every element, so the branch
// costs nothing. Rank four, any extents.

PG_ABI void pgAxpby(int count, const pgView *A_, double a, double b,
                    const pgView *B_) {
  for (int e = 0; e < count; e++) {
    auto A = as4(A_[e]), B = as4(B_[e]);
    MDRange4 range({0, 0, 0, 0},
                   {A.extent(0), A.extent(1), A.extent(2), A.extent(3)});
    Kokkos::parallel_for(
        "axpby", range,
        KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
          A(i, j, k, l) = a == 0.0 ? b * B(i, j, k, l)
                                   : a * A(i, j, k, l) + b * B(i, j, k, l);
        });
  }
}
