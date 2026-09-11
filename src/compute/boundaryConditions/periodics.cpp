#include "block_.hpp"
#include "compute.hpp"
#include "face_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <Kokkos_Core.hpp>
#include <string.h>

void periodicRot(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> /*&eos*/,
    const thtrdat_ /*&th*/, const std::string &terms, const double /*&tme*/) {
  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const int ng = b.ng;
  int firstHaloIdx, firstInteriorCellIdx, blockFaceIdx, plus;
  getFaceSliceIdxs(firstHaloIdx, firstInteriorCellIdx, blockFaceIdx, plus, b.ni,
                   b.nj, b.nk, ng, face.nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face.nface, firstInteriorCellIdx);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face.nface, firstHaloIdx);
      threeDsubview Q0 = getFaceSlice(b.Q, face.nface, firstHaloIdx);

      Kokkos::parallel_for(
          "Rotate periodic euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // turn the velocity vector onto this face
            double tempU, tempV, tempW;
            double u = q0(i, j, 1);
            double v = q0(i, j, 2);
            double w = q0(i, j, 3);
            tempU = face.periodicRotMatrix(0, 0) * u +
                    face.periodicRotMatrix(0, 1) * v +
                    face.periodicRotMatrix(0, 2) * w;
            tempV = face.periodicRotMatrix(1, 0) * u +
                    face.periodicRotMatrix(1, 1) * v +
                    face.periodicRotMatrix(1, 2) * w;
            tempW = face.periodicRotMatrix(2, 0) * u +
                    face.periodicRotMatrix(2, 1) * v +
                    face.periodicRotMatrix(2, 2) * w;

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

    fourDsubview grads0 = getFaceSlice(b.grads, face.nface, firstHaloIdx);

    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(grads0.extent(0)),
                             static_cast<long>(grads0.extent(1)), b.ne});
    Kokkos::parallel_for(
        "Periodic postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j, const int l) {
          // turn the gradient vectors onto this face
          double grad[3] = {grads0(i, j, l, 0), grads0(i, j, l, 1),
                            grads0(i, j, l, 2)};
          for (int r = 0; r < 3; r++) {
            double turned = 0.0;
            for (int c = 0; c < 3; c++) {
              turned += face.periodicRotMatrix(r, c) * grad[c];
            }
            grads0(i, j, l, r) = turned;
          }
        });
  }
}
