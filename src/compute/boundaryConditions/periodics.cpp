#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include "face_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <string.h>

void periodicRotHigh(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> /*&eos*/,
    const thtrdat_ /*&th*/, const std::string &terms, const double /*&tme*/) {
  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const int ng = b.ng;
  int firstHaloIdx, firstInteriorCellIdx, blockFaceIdx, plus;
  getFaceSliceIdxs(firstHaloIdx, firstInteriorCellIdx, blockFaceIdx, plus, b.ni,
                   b.nj, b.nk, ng, face._nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face._nface, firstInteriorCellIdx);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face._nface, firstHaloIdx);
      threeDsubview Q0 = getFaceSlice(b.Q, face._nface, firstHaloIdx);

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
  } else if (terms.compare("postDqDxyz") == 0) {

    threeDsubview dqdx1 =
        getFaceSlice(b.dqdx, face._nface, firstInteriorCellIdx);
    threeDsubview dqdx0 = getFaceSlice(b.dqdx, face._nface, firstHaloIdx);
    threeDsubview dqdy0 = getFaceSlice(b.dqdy, face._nface, firstHaloIdx);
    threeDsubview dqdz0 = getFaceSlice(b.dqdz, face._nface, firstHaloIdx);

    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(dqdx1.extent(0)),
                             static_cast<long>(dqdx1.extent(1)), b.ne});
    Kokkos::parallel_for(
        "Periodic postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j, const int l) {
          // rotate gradients vector up to high face
          double tempdx, tempdy, tempdz;
          double dx = dqdx0(i, j, l);
          double dy = dqdy0(i, j, l);
          double dz = dqdz0(i, j, l);
          tempdx = face.periodicRotMatrixUp(0, 0) * dx +
                   face.periodicRotMatrixUp(0, 1) * dy +
                   face.periodicRotMatrixUp(0, 2) * dz;
          tempdy = face.periodicRotMatrixUp(1, 0) * dx +
                   face.periodicRotMatrixUp(1, 1) * dy +
                   face.periodicRotMatrixUp(1, 2) * dz;
          tempdz = face.periodicRotMatrixUp(2, 0) * dx +
                   face.periodicRotMatrixUp(2, 1) * dy +
                   face.periodicRotMatrixUp(2, 2) * dz;

          dqdx0(i, j, l) = tempdx;
          dqdy0(i, j, l) = tempdy;
          dqdz0(i, j, l) = tempdz;
        });
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
  int firstHaloIdx, firstInteriorCellIdx, blockFaceIdx, plus;
  getFaceSliceIdxs(firstHaloIdx, firstInteriorCellIdx, blockFaceIdx, plus, b.ni,
                   b.nj, b.nk, ng, face._nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face._nface, firstInteriorCellIdx);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face._nface, firstHaloIdx);
      threeDsubview Q0 = getFaceSlice(b.Q, face._nface, firstHaloIdx);

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
  } else if (terms.compare("postDqDxyz") == 0) {

    threeDsubview dqdx1 =
        getFaceSlice(b.dqdx, face._nface, firstInteriorCellIdx);
    threeDsubview dqdx0 = getFaceSlice(b.dqdx, face._nface, firstHaloIdx);
    threeDsubview dqdy0 = getFaceSlice(b.dqdy, face._nface, firstHaloIdx);
    threeDsubview dqdz0 = getFaceSlice(b.dqdz, face._nface, firstHaloIdx);

    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(dqdx1.extent(0)),
                             static_cast<long>(dqdx1.extent(1)), b.ne});
    Kokkos::parallel_for(
        "Periodic postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j, const int l) {
          // rotate gradients vector up to high face
          double tempdx, tempdy, tempdz;
          double dx = dqdx0(i, j, l);
          double dy = dqdy0(i, j, l);
          double dz = dqdz0(i, j, l);
          tempdx = face.periodicRotMatrixDown(0, 0) * dx +
                   face.periodicRotMatrixDown(0, 1) * dy +
                   face.periodicRotMatrixDown(0, 2) * dz;
          tempdy = face.periodicRotMatrixDown(1, 0) * dx +
                   face.periodicRotMatrixDown(1, 1) * dy +
                   face.periodicRotMatrixDown(1, 2) * dz;
          tempdz = face.periodicRotMatrixDown(2, 0) * dx +
                   face.periodicRotMatrixDown(2, 1) * dy +
                   face.periodicRotMatrixDown(2, 2) * dz;

          dqdx0(i, j, l) = tempdx;
          dqdy0(i, j, l) = tempdy;
          dqdz0(i, j, l) = tempdz;
        });
  }
}
