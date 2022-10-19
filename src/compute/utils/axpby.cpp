#include "Kokkos_Core.hpp"
#include "kokkosTypes.hpp"

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
  int indxI = A.extent(0);
  int indxJ = A.extent(1);
  int indxK = A.extent(2);
  int indxL = A.extent(3);
  MDRange4 range({0, 0, 0, 0}, {indxI, indxJ, indxK, indxL});
  Kokkos::parallel_for(
      "AEQB", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        A(i, j, k, l) = B(i, j, k, l);
      });
}

void ApEQxB(fourDview &A, const double &x, fourDview &B) {
  //-------------------------------------------------------------------------------------------|
  // A += x*B
  //-------------------------------------------------------------------------------------------|
  int indxI = A.extent(0);
  int indxJ = A.extent(1);
  int indxK = A.extent(2);
  int indxL = A.extent(3);
  MDRange4 range({0, 0, 0, 0}, {indxI, indxJ, indxK, indxL});
  Kokkos::parallel_for(
      "ApEQxB", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        A(i, j, k, l) += x * B(i, j, k, l);
      });
}

void AEQxB(fourDview &A, const double &x, fourDview &B) {
  //-------------------------------------------------------------------------------------------|
  // A = xB
  //-------------------------------------------------------------------------------------------|
  int indxI = A.extent(0);
  int indxJ = A.extent(1);
  int indxK = A.extent(2);
  int indxL = A.extent(3);
  MDRange4 range({0, 0, 0, 0}, {indxI, indxJ, indxK, indxL});
  Kokkos::parallel_for(
      "AEQxb", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        A(i, j, k, l) = x * B(i, j, k, l);
      });
}

void CEQxApyB(fourDview &C, const double &x, const fourDview &A,
              const double &y, const fourDview &B) {
  //-------------------------------------------------------------------------------------------|
  // C = Ax + By
  //-------------------------------------------------------------------------------------------|
  int indxI = A.extent(0);
  int indxJ = A.extent(1);
  int indxK = A.extent(2);
  int indxL = A.extent(3);
  MDRange4 range({0, 0, 0, 0}, {indxI, indxJ, indxK, indxL});
  Kokkos::parallel_for(
      "CEQxApyB", range,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        C(i, j, k, l) = x * A(i, j, k, l) + y * B(i, j, k, l);
      });
}
