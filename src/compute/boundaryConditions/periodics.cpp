#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include "face_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"

void periodicRotHigh(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> /*&eos*/,
    const thtrdat_ /*&th*/, const std::string &terms, const double /*&tme*/) {
  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const int ng = b.ng;
  int s0, s1, s2, plus;
  setHaloSlices(s0, s1, s2, plus, b.ni, b.nj, b.nk, ng, face._nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getHaloSlice(b.q, face._nface, s1);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);
      threeDsubview Q0 = getHaloSlice(b.Q, face._nface, s0);

      Kokkos::parallel_for(
          "Rotate periodic euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // rotate velocity vector up to high face
            double tempU, tempV, tempW;
            double u = q0(i, j, 1);
            double v = q0(i, j, 2);
            double w = q0(i, j, 3);
            tempU = face.periodicRotMatrixUp(0, 0) * u +
                    face.periodicRotMatrixUp(0, 1) * v +
                    face.periodicRotMatrixUp(0, 2) * w;
            tempV = face.periodicRotMatrixUp(1, 0) * u +
                    face.periodicRotMatrixUp(1, 1) * v +
                    face.periodicRotMatrixUp(1, 2) * w;
            tempW = face.periodicRotMatrixUp(2, 0) * u +
                    face.periodicRotMatrixUp(2, 1) * v +
                    face.periodicRotMatrixUp(2, 2) * w;

            // Update velocity
            q0(i, j, 1) = tempU;
            q0(i, j, 2) = tempV;
            q0(i, j, 3) = tempW;

            // Update momentum
            Q0(i, j, 1) = tempU * Q0(i, j, 0);
            Q0(i, j, 2) = tempV * Q0(i, j, 0);
            Q0(i, j, 3) = tempW * Q0(i, j, 0);
          });
    }
  }
}

void periodicRotLow(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> /*&eos*/,
    const thtrdat_ /*&th*/, const std::string &terms, const double /*&tme*/) {
  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const int ng = b.ng;
  int s0, s1, s2, plus;
  setHaloSlices(s0, s1, s2, plus, b.ni, b.nj, b.nk, ng, face._nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getHaloSlice(b.q, face._nface, s1);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);
      threeDsubview Q0 = getHaloSlice(b.Q, face._nface, s0);

      Kokkos::parallel_for(
          "Rotate periodic euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // rotate velocity vector up to high face
            double tempU, tempV, tempW;
            double u = q0(i, j, 1);
            double v = q0(i, j, 2);
            double w = q0(i, j, 3);
            tempU = face.periodicRotMatrixDown(0, 0) * u +
                    face.periodicRotMatrixDown(0, 1) * v +
                    face.periodicRotMatrixDown(0, 2) * w;
            tempV = face.periodicRotMatrixDown(1, 0) * u +
                    face.periodicRotMatrixDown(1, 1) * v +
                    face.periodicRotMatrixDown(1, 2) * w;
            tempW = face.periodicRotMatrixDown(2, 0) * u +
                    face.periodicRotMatrixDown(2, 1) * v +
                    face.periodicRotMatrixDown(2, 2) * w;

            // Update velocity
            q0(i, j, 1) = tempU;
            q0(i, j, 2) = tempV;
            q0(i, j, 3) = tempW;

            // Update momentum
            Q0(i, j, 1) = tempU * Q0(i, j, 0);
            Q0(i, j, 2) = tempV * Q0(i, j, 0);
            Q0(i, j, 3) = tempW * Q0(i, j, 0);
          });
    }
  }
}
