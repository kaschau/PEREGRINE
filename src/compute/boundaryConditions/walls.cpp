#include "block_.hpp"
#include "compute.hpp"
#include "face_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <Kokkos_Core.hpp>
#include <string.h>

// Strategy for wall halo velocities:
//  For euler boundary conditions, we make all walls
//  slip walls. This is for computation of inviscid fluxes.
//  Then we apply the viscous bcs ("preDqDxyz")and make no
//  slip walls correct, so that velocity gradients will be correct
//  on no slip wall faces. After gradients ("postDqDxyz") we apply
//  the velocity gradients in the halos to have desired effect.

void adiabaticNoSlipWall(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ &th, const std::string &terms, const double /*&tme*/) {
  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const int ng = b.ng;
  int firstHaloIdx, firstInteriorCellIdx, blockFaceIdx, plus;
  getFaceSliceIdxs(firstHaloIdx, firstInteriorCellIdx, blockFaceIdx, plus, b.ni,
                   b.nj, b.nk, ng, face.nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face.nface, firstInteriorCellIdx);
    threeDsubview sVec = getFaceAreaVectors(b, face.nface, blockFaceIdx);

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;
      threeDsubview q0 = getFaceSlice(b.q, face.nface, firstHaloIdx);

      Kokkos::parallel_for(
          "Adia no slip wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            double S, nx, ny, nz;
            faceNormal(sVec(i, j, 0), sVec(i, j, 1), sVec(i, j, 2), S, nx, ny,
                       nz);

            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn =
                q1(i, j, 1) * nx + q1(i, j, 2) * ny + q1(i, j, 3) * nz;
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz;

            // match temperature
            q0(i, j, 4) = q1(i, j, 4);
            // match species
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }
    eos(b, th, face.nface, "prims");

  } else if (terms.compare("preDqDxyz") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face.nface, firstInteriorCellIdx);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face.nface, firstHaloIdx);

      Kokkos::parallel_for(
          "Adia no slip wall preDqDxyz terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // flip velo on wall
            q0(i, j, 1) = -q1(i, j, 1);
            q0(i, j, 2) = -q1(i, j, 2);
            q0(i, j, 3) = -q1(i, j, 3);
          });
    }
  } else if (terms.compare("postDqDxyz") == 0) {

    // Only applied to first halo slice
    fourDsubview grads0 = getFaceSlice(b.grads, face.nface, firstHaloIdx);

    fourDsubview grads1 =
        getFaceSlice(b.grads, face.nface, firstInteriorCellIdx);

    MDRange2 range_face =
        MDRange2({0, 0}, {grads1.extent(0), grads1.extent(1)});
    Kokkos::parallel_for(
        "Adia no slip postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          // negate pressure,  neumann velocity gradients
          grads0(i, j, 0, 0) = -grads1(i, j, 0, 0);
          grads0(i, j, 1, 0) = grads1(i, j, 1, 0);
          grads0(i, j, 2, 0) = grads1(i, j, 2, 0);
          grads0(i, j, 3, 0) = grads1(i, j, 3, 0);

          grads0(i, j, 0, 1) = -grads1(i, j, 0, 1);
          grads0(i, j, 1, 1) = grads1(i, j, 1, 1);
          grads0(i, j, 2, 1) = grads1(i, j, 2, 1);
          grads0(i, j, 3, 1) = grads1(i, j, 3, 1);

          grads0(i, j, 0, 2) = -grads1(i, j, 0, 2);
          grads0(i, j, 1, 2) = grads1(i, j, 1, 2);
          grads0(i, j, 2, 2) = grads1(i, j, 2, 2);
          grads0(i, j, 3, 2) = grads1(i, j, 3, 2);

          // negate temp and species gradient (so gradient evaluates to zero
          // on wall)
          grads0(i, j, 4, 0) = -grads1(i, j, 4, 0);
          grads0(i, j, 4, 1) = -grads1(i, j, 4, 1);
          grads0(i, j, 4, 2) = -grads1(i, j, 4, 2);

          for (int n = 5; n < b.ne; n++) {
            grads0(i, j, n, 0) = -grads1(i, j, n, 0);
            grads0(i, j, n, 1) = -grads1(i, j, n, 1);
            grads0(i, j, n, 2) = -grads1(i, j, n, 2);
          }
        });
  }
}

void adiabaticSlipWall(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ &th, const std::string &terms, const double /*&tme*/) {
  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const int ng = b.ng;
  int firstHaloIdx, firstInteriorCellIdx, blockFaceIdx, plus;
  getFaceSliceIdxs(firstHaloIdx, firstInteriorCellIdx, blockFaceIdx, plus, b.ni,
                   b.nj, b.nk, ng, face.nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face.nface, firstInteriorCellIdx);
    threeDsubview sVec = getFaceAreaVectors(b, face.nface, blockFaceIdx);

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face.nface, firstHaloIdx);

      Kokkos::parallel_for(
          "Adia slip wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            double S, nx, ny, nz;
            faceNormal(sVec(i, j, 0), sVec(i, j, 1), sVec(i, j, 2), S, nx, ny,
                       nz);

            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn =
                q1(i, j, 1) * nx + q1(i, j, 2) * ny + q1(i, j, 3) * nz;
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz;

            // match temperature
            q0(i, j, 4) = q1(i, j, 4);
            // match species
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }
    eos(b, th, face.nface, "prims");
  } else if (terms.compare("postDqDxyz") == 0) {

    // Only applied to first halo slice
    fourDsubview grads0 = getFaceSlice(b.grads, face.nface, firstHaloIdx);

    fourDsubview grads1 =
        getFaceSlice(b.grads, face.nface, firstInteriorCellIdx);

    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(grads1.extent(0)),
                             static_cast<long>(grads1.extent(1)), b.ne});
    Kokkos::parallel_for(
        "Adia slip visc terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j, const int l) {
          // negate all gradients
          for (int d = 0; d < 3; d++) {
            grads0(i, j, l, d) = -grads1(i, j, l, d);
          }
        });
  }
}

void adiabaticMovingWall(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ &th, const std::string &terms, const double /*&tme*/) {
  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const int ng = b.ng;
  int firstHaloIdx, firstInteriorCellIdx, blockFaceIdx, plus;
  getFaceSliceIdxs(firstHaloIdx, firstInteriorCellIdx, blockFaceIdx, plus, b.ni,
                   b.nj, b.nk, ng, face.nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face.nface, firstInteriorCellIdx);
    threeDsubview sVec = getFaceAreaVectors(b, face.nface, blockFaceIdx);

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face.nface, firstHaloIdx);

      Kokkos::parallel_for(
          "Adia moving wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            double S, nx, ny, nz;
            faceNormal(sVec(i, j, 0), sVec(i, j, 1), sVec(i, j, 2), S, nx, ny,
                       nz);

            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn =
                q1(i, j, 1) * nx + q1(i, j, 2) * ny + q1(i, j, 3) * nz;
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz;

            // match temperature
            q0(i, j, 4) = q1(i, j, 4);
            // match species
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }
    eos(b, th, face.nface, "prims");

  } else if (terms.compare("preDqDxyz") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face.nface, firstInteriorCellIdx);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face.nface, firstHaloIdx);
      Kokkos::parallel_for(
          "Adia moving wall preDqDxyz terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // apply velo to face
            q0(i, j, 1) = 2.0 * face.qBcVals(i, j, 1) - q1(i, j, 1);
            q0(i, j, 2) = 2.0 * face.qBcVals(i, j, 2) - q1(i, j, 2);
            q0(i, j, 3) = 2.0 * face.qBcVals(i, j, 3) - q1(i, j, 3);
          });
    }
  } else if (terms.compare("postDqDxyz") == 0) {

    fourDsubview grads0 = getFaceSlice(b.grads, face.nface, firstHaloIdx);

    fourDsubview grads1 =
        getFaceSlice(b.grads, face.nface, firstInteriorCellIdx);

    MDRange2 range_face =
        MDRange2({0, 0}, {grads1.extent(0), grads1.extent(1)});
    Kokkos::parallel_for(
        "Adia moving wall postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          // negate pressure,  neumann velocity gradients
          grads0(i, j, 0, 0) = -grads1(i, j, 0, 0);
          grads0(i, j, 1, 0) = grads1(i, j, 1, 0);
          grads0(i, j, 2, 0) = grads1(i, j, 2, 0);
          grads0(i, j, 3, 0) = grads1(i, j, 3, 0);

          grads0(i, j, 0, 1) = -grads1(i, j, 0, 1);
          grads0(i, j, 1, 1) = grads1(i, j, 1, 1);
          grads0(i, j, 2, 1) = grads1(i, j, 2, 1);
          grads0(i, j, 3, 1) = grads1(i, j, 3, 1);

          grads0(i, j, 0, 2) = -grads1(i, j, 0, 2);
          grads0(i, j, 1, 2) = grads1(i, j, 1, 2);
          grads0(i, j, 2, 2) = grads1(i, j, 2, 2);
          grads0(i, j, 3, 2) = grads1(i, j, 3, 2);

          // negate temp and species gradient (so gradient evaluates to zero
          // on wall)
          grads0(i, j, 4, 0) = -grads1(i, j, 4, 0);
          grads0(i, j, 4, 1) = -grads1(i, j, 4, 1);
          grads0(i, j, 4, 2) = -grads1(i, j, 4, 2);

          for (int n = 5; n < b.ne; n++) {
            grads0(i, j, n, 0) = -grads1(i, j, n, 0);
            grads0(i, j, n, 1) = -grads1(i, j, n, 1);
            grads0(i, j, n, 2) = -grads1(i, j, n, 2);
          }
        });
  }
}

void isoTNoSlipWall(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ &th, const std::string &terms, const double /*&tme*/) {
  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const int ng = b.ng;
  int firstHaloIdx, firstInteriorCellIdx, blockFaceIdx, plus;
  getFaceSliceIdxs(firstHaloIdx, firstInteriorCellIdx, blockFaceIdx, plus, b.ni,
                   b.nj, b.nk, ng, face.nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face.nface, firstInteriorCellIdx);
    threeDsubview sVec = getFaceAreaVectors(b, face.nface, blockFaceIdx);

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face.nface, firstHaloIdx);

      Kokkos::parallel_for(
          "isoT no slip wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            double S, nx, ny, nz;
            faceNormal(sVec(i, j, 0), sVec(i, j, 1), sVec(i, j, 2), S, nx, ny,
                       nz);

            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn =
                q1(i, j, 1) * nx + q1(i, j, 2) * ny + q1(i, j, 3) * nz;
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz;

            // set temperature
            q0(i, j, 4) = face.qBcVals(i, j, 4);
            // match species
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }
    eos(b, th, face.nface, "prims");

  } else if (terms.compare("preDqDxyz") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face.nface, firstInteriorCellIdx);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face.nface, firstHaloIdx);

      Kokkos::parallel_for(
          "isoT no slip wall preDqDxyz terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // flip velo on wall
            q0(i, j, 1) = -q1(i, j, 1);
            q0(i, j, 2) = -q1(i, j, 2);
            q0(i, j, 3) = -q1(i, j, 3);
          });
    }
  } else if (terms.compare("postDqDxyz") == 0) {

    fourDsubview grads0 = getFaceSlice(b.grads, face.nface, firstHaloIdx);

    fourDsubview grads1 =
        getFaceSlice(b.grads, face.nface, firstInteriorCellIdx);

    MDRange2 range_face =
        MDRange2({0, 0}, {grads1.extent(0), grads1.extent(1)});
    Kokkos::parallel_for(
        "isoT no slip postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          // negate pressure,  neumann velocity,temperature gradients
          grads0(i, j, 0, 0) = -grads1(i, j, 0, 0);
          grads0(i, j, 1, 0) = grads1(i, j, 1, 0);
          grads0(i, j, 2, 0) = grads1(i, j, 2, 0);
          grads0(i, j, 3, 0) = grads1(i, j, 3, 0);
          grads0(i, j, 4, 0) = grads1(i, j, 4, 0);

          grads0(i, j, 0, 1) = -grads1(i, j, 0, 1);
          grads0(i, j, 1, 1) = grads1(i, j, 1, 1);
          grads0(i, j, 2, 1) = grads1(i, j, 2, 1);
          grads0(i, j, 3, 1) = grads1(i, j, 3, 1);
          grads0(i, j, 4, 1) = grads1(i, j, 4, 1);

          grads0(i, j, 0, 2) = -grads1(i, j, 0, 2);
          grads0(i, j, 1, 2) = grads1(i, j, 1, 2);
          grads0(i, j, 2, 2) = grads1(i, j, 2, 2);
          grads0(i, j, 3, 2) = grads1(i, j, 3, 2);
          grads0(i, j, 4, 2) = grads1(i, j, 4, 2);

          // negatespecies gradient (so gradient evaluates to zero
          // on wall)
          for (int n = 5; n < b.ne; n++) {
            grads0(i, j, n, 0) = -grads1(i, j, n, 0);
            grads0(i, j, n, 1) = -grads1(i, j, n, 1);
            grads0(i, j, n, 2) = -grads1(i, j, n, 2);
          }
        });
  }
}

void isoTSlipWall(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ &th, const std::string &terms, const double /*&tme*/) {
  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const int ng = b.ng;
  int firstHaloIdx, firstInteriorCellIdx, blockFaceIdx, plus;
  getFaceSliceIdxs(firstHaloIdx, firstInteriorCellIdx, blockFaceIdx, plus, b.ni,
                   b.nj, b.nk, ng, face.nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face.nface, firstInteriorCellIdx);
    threeDsubview sVec = getFaceAreaVectors(b, face.nface, blockFaceIdx);

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face.nface, firstHaloIdx);

      Kokkos::parallel_for(
          "isoT slip wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            double S, nx, ny, nz;
            faceNormal(sVec(i, j, 0), sVec(i, j, 1), sVec(i, j, 2), S, nx, ny,
                       nz);

            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // flip velo on wall
            double uDotn =
                q1(i, j, 1) * nx + q1(i, j, 2) * ny + q1(i, j, 3) * nz;
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz;

            // set temperature
            q0(i, j, 4) = face.qBcVals(i, j, 4);
            // match species
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }
    eos(b, th, face.nface, "prims");

  } else if (terms.compare("postDqDxyz") == 0) {

    fourDsubview grads0 = getFaceSlice(b.grads, face.nface, firstHaloIdx);

    fourDsubview grads1 =
        getFaceSlice(b.grads, face.nface, firstInteriorCellIdx);

    MDRange2 range_face =
        MDRange2({0, 0}, {grads1.extent(0), grads1.extent(1)});
    Kokkos::parallel_for(
        "isoT slip visc terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          // negate velocity gradients
          grads0(i, j, 0, 0) = -grads1(i, j, 0, 0);
          grads0(i, j, 1, 0) = -grads1(i, j, 1, 0);
          grads0(i, j, 2, 0) = -grads1(i, j, 2, 0);
          grads0(i, j, 3, 0) = -grads1(i, j, 3, 0);

          grads0(i, j, 0, 1) = -grads1(i, j, 0, 1);
          grads0(i, j, 1, 1) = -grads1(i, j, 1, 1);
          grads0(i, j, 2, 1) = -grads1(i, j, 2, 1);
          grads0(i, j, 3, 1) = -grads1(i, j, 3, 1);

          grads0(i, j, 0, 2) = -grads1(i, j, 0, 2);
          grads0(i, j, 1, 2) = -grads1(i, j, 1, 2);
          grads0(i, j, 2, 2) = -grads1(i, j, 2, 2);
          grads0(i, j, 3, 2) = -grads1(i, j, 3, 2);

          // neumann temp gradients
          grads0(i, j, 4, 0) = grads1(i, j, 4, 0);
          grads0(i, j, 4, 1) = grads1(i, j, 4, 1);
          grads0(i, j, 4, 2) = grads1(i, j, 4, 2);

          // negate species gradient (so gradient evaluates to zero
          // on wall)
          for (int n = 5; n < b.ne; n++) {
            grads0(i, j, n, 0) = -grads1(i, j, n, 0);
            grads0(i, j, n, 1) = -grads1(i, j, n, 1);
            grads0(i, j, n, 2) = -grads1(i, j, n, 2);
          }
        });
  }
}

void isoTMovingWall(
    block_ &b, face_ &face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ &th, const std::string &terms, const double /*&tme*/) {
  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const int ng = b.ng;
  int firstHaloIdx, firstInteriorCellIdx, blockFaceIdx, plus;
  getFaceSliceIdxs(firstHaloIdx, firstInteriorCellIdx, blockFaceIdx, plus, b.ni,
                   b.nj, b.nk, ng, face.nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face.nface, firstInteriorCellIdx);
    threeDsubview sVec = getFaceAreaVectors(b, face.nface, blockFaceIdx);

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face.nface, firstHaloIdx);

      Kokkos::parallel_for(
          "Iso T moving wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            double S, nx, ny, nz;
            faceNormal(sVec(i, j, 0), sVec(i, j, 1), sVec(i, j, 2), S, nx, ny,
                       nz);

            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn =
                q1(i, j, 1) * nx + q1(i, j, 2) * ny + q1(i, j, 3) * nz;
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz;

            // set temperature
            q0(i, j, 4) = face.qBcVals(i, j, 4);
            // match species
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }
    eos(b, th, face.nface, "prims");

  } else if (terms.compare("preDqDxyz") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face.nface, firstInteriorCellIdx);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face.nface, firstHaloIdx);

      Kokkos::parallel_for(
          "Iso T moving wall preDqDxyz terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // apply velo on wall
            q0(i, j, 1) = 2.0 * face.qBcVals(i, j, 1) - q1(i, j, 1);
            q0(i, j, 2) = 2.0 * face.qBcVals(i, j, 2) - q1(i, j, 2);
            q0(i, j, 3) = 2.0 * face.qBcVals(i, j, 3) - q1(i, j, 3);
          });
    }
  } else if (terms.compare("postDqDxyz") == 0) {

    fourDsubview grads0 = getFaceSlice(b.grads, face.nface, firstHaloIdx);

    fourDsubview grads1 =
        getFaceSlice(b.grads, face.nface, firstInteriorCellIdx);

    MDRange2 range_face =
        MDRange2({0, 0}, {grads1.extent(0), grads1.extent(1)});
    Kokkos::parallel_for(
        "Iso T moving wall postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          // negate pressure gradient, neumann velocity, temperature gradients
          grads0(i, j, 0, 0) = -grads1(i, j, 0, 0);
          grads0(i, j, 1, 0) = grads1(i, j, 1, 0);
          grads0(i, j, 2, 0) = grads1(i, j, 2, 0);
          grads0(i, j, 3, 0) = grads1(i, j, 3, 0);
          grads0(i, j, 4, 0) = grads1(i, j, 4, 0);

          grads0(i, j, 0, 1) = -grads1(i, j, 0, 1);
          grads0(i, j, 1, 1) = grads1(i, j, 1, 1);
          grads0(i, j, 2, 1) = grads1(i, j, 2, 1);
          grads0(i, j, 3, 1) = grads1(i, j, 3, 1);
          grads0(i, j, 4, 1) = grads1(i, j, 4, 1);

          grads0(i, j, 0, 2) = -grads1(i, j, 0, 2);
          grads0(i, j, 1, 2) = grads1(i, j, 1, 2);
          grads0(i, j, 2, 2) = grads1(i, j, 2, 2);
          grads0(i, j, 3, 2) = grads1(i, j, 3, 2);
          grads0(i, j, 4, 2) = grads1(i, j, 4, 2);

          // negate species gradient (so gradient evaluates to zero on wall)
          for (int n = 5; n < b.ne; n++) {
            grads0(i, j, n, 0) = -grads1(i, j, n, 0);
            grads0(i, j, n, 1) = -grads1(i, j, n, 1);
            grads0(i, j, n, 2) = -grads1(i, j, n, 2);
          }
        });
  }
}
