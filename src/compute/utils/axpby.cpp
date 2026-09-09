#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

void AEQConst(fourDview &A, const double &Const) {
  //-------------------------------------------------------------------------------------------|
  // A = Const
  //-------------------------------------------------------------------------------------------|
  Kokkos::deep_copy(A, Const);
}

void AEQConst(threeDview &A, const double &Const) {
  //-------------------------------------------------------------------------------------------|
  // A = Const
  //-------------------------------------------------------------------------------------------|
  Kokkos::deep_copy(A, Const);
}

void AEQB(fourDview &A, fourDview &B) {
  //-------------------------------------------------------------------------------------------|
  // A = B
  //-------------------------------------------------------------------------------------------|
  MDRange4 range({0, 0, 0, 0},
                 {A.extent(0), A.extent(1), A.extent(2), A.extent(3)});
  Kokkos::parallel_for(
      "AEQB", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        A(i, j, k, l) = B(i, j, k, l);
      });
}

//-------------------------------------------------------------------------------------------|
// A = a*A + b*B [+ c*C]
//
// The linear combination of solution registers every time integration stage is
// built from. A leading coefficient of zero means the stage starts from
// somewhere else, so A is written without being read; the test is on a value
// that is the same for every element, so the branch costs nothing.
//-------------------------------------------------------------------------------------------|
void axnpby(fourDview &A, const double &a, const double &b,
            const fourDview &B) {
  MDRange4 range({0, 0, 0, 0},
                 {A.extent(0), A.extent(1), A.extent(2), A.extent(3)});
  Kokkos::parallel_for(
      "axnpby2", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        if (a == 0.0) {
          A(i, j, k, l) = b * B(i, j, k, l);
        } else {
          A(i, j, k, l) = a * A(i, j, k, l) + b * B(i, j, k, l);
        }
      });
}

void axnpby(fourDview &A, const double &a, const double &b, const fourDview &B,
            const double &c, const fourDview &C) {
  MDRange4 range({0, 0, 0, 0},
                 {A.extent(0), A.extent(1), A.extent(2), A.extent(3)});
  Kokkos::parallel_for(
      "axnpby3", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        if (a == 0.0) {
          A(i, j, k, l) = b * B(i, j, k, l) + c * C(i, j, k, l);
        } else {
          A(i, j, k, l) =
              a * A(i, j, k, l) + b * B(i, j, k, l) + c * C(i, j, k, l);
        }
      });
}
