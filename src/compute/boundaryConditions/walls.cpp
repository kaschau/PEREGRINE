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
                   b.nj, b.nk, ng, face._nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face._nface, firstInteriorCellIdx);
    twoDsubview nx, ny, nz;

    switch (face._nface) {
    case 1:
    case 2:
      nx = getFaceSlice(b.inx, face._nface, blockFaceIdx);
      ny = getFaceSlice(b.iny, face._nface, blockFaceIdx);
      nz = getFaceSlice(b.inz, face._nface, blockFaceIdx);
      break;
    case 3:
    case 4:
      nx = getFaceSlice(b.jnx, face._nface, blockFaceIdx);
      ny = getFaceSlice(b.jny, face._nface, blockFaceIdx);
      nz = getFaceSlice(b.jnz, face._nface, blockFaceIdx);
      break;
    case 5:
    case 6:
      nx = getFaceSlice(b.knx, face._nface, blockFaceIdx);
      ny = getFaceSlice(b.kny, face._nface, blockFaceIdx);
      nz = getFaceSlice(b.knz, face._nface, blockFaceIdx);
      break;
    }

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;
      threeDsubview q0 = getFaceSlice(b.q, face._nface, firstHaloIdx);

      Kokkos::parallel_for(
          "Adia no slip wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn = q1(i, j, 1) * nx(i, j) + q1(i, j, 2) * ny(i, j) +
                           q1(i, j, 3) * nz(i, j);
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx(i, j);
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny(i, j);
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz(i, j);

            // match temperature
            q0(i, j, 4) = q1(i, j, 4);
            // match species
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }
    eos(b, th, face._nface, "prims");

  } else if (terms.compare("preDqDxyz") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face._nface, firstInteriorCellIdx);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face._nface, firstHaloIdx);

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
    threeDsubview dqdx0 = getFaceSlice(b.dqdx, face._nface, firstHaloIdx);
    threeDsubview dqdy0 = getFaceSlice(b.dqdy, face._nface, firstHaloIdx);
    threeDsubview dqdz0 = getFaceSlice(b.dqdz, face._nface, firstHaloIdx);

    threeDsubview dqdx1 =
        getFaceSlice(b.dqdx, face._nface, firstInteriorCellIdx);
    threeDsubview dqdy1 =
        getFaceSlice(b.dqdy, face._nface, firstInteriorCellIdx);
    threeDsubview dqdz1 =
        getFaceSlice(b.dqdz, face._nface, firstInteriorCellIdx);

    MDRange2 range_face = MDRange2({0, 0}, {dqdx1.extent(0), dqdx1.extent(1)});
    Kokkos::parallel_for(
        "Adia no slip postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          // negate pressure,  neumann velocity gradients
          dqdx0(i, j, 0) = -dqdx1(i, j, 0);
          dqdx0(i, j, 1) = dqdx1(i, j, 1);
          dqdx0(i, j, 2) = dqdx1(i, j, 2);
          dqdx0(i, j, 3) = dqdx1(i, j, 3);

          dqdy0(i, j, 0) = -dqdy1(i, j, 0);
          dqdy0(i, j, 1) = dqdy1(i, j, 1);
          dqdy0(i, j, 2) = dqdy1(i, j, 2);
          dqdy0(i, j, 3) = dqdy1(i, j, 3);

          dqdz0(i, j, 0) = -dqdz1(i, j, 0);
          dqdz0(i, j, 1) = dqdz1(i, j, 1);
          dqdz0(i, j, 2) = dqdz1(i, j, 2);
          dqdz0(i, j, 3) = dqdz1(i, j, 3);

          // negate temp and species gradient (so gradient evaluates to zero
          // on wall)
          dqdx0(i, j, 4) = -dqdx1(i, j, 4);
          dqdy0(i, j, 4) = -dqdy1(i, j, 4);
          dqdz0(i, j, 4) = -dqdz1(i, j, 4);

          for (int n = 5; n < b.ne; n++) {
            dqdx0(i, j, n) = -dqdx1(i, j, n);
            dqdy0(i, j, n) = -dqdy1(i, j, n);
            dqdz0(i, j, n) = -dqdz1(i, j, n);
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
                   b.nj, b.nk, ng, face._nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face._nface, firstInteriorCellIdx);
    twoDsubview nx, ny, nz;
    switch (face._nface) {
    case 1:
    case 2:
      nx = getFaceSlice(b.inx, face._nface, blockFaceIdx);
      ny = getFaceSlice(b.iny, face._nface, blockFaceIdx);
      nz = getFaceSlice(b.inz, face._nface, blockFaceIdx);
      break;
    case 3:
    case 4:
      nx = getFaceSlice(b.jnx, face._nface, blockFaceIdx);
      ny = getFaceSlice(b.jny, face._nface, blockFaceIdx);
      nz = getFaceSlice(b.jnz, face._nface, blockFaceIdx);
      break;
    case 5:
    case 6:
      nx = getFaceSlice(b.knx, face._nface, blockFaceIdx);
      ny = getFaceSlice(b.kny, face._nface, blockFaceIdx);
      nz = getFaceSlice(b.knz, face._nface, blockFaceIdx);
      break;
    }

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face._nface, firstHaloIdx);

      Kokkos::parallel_for(
          "Adia slip wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn = q1(i, j, 1) * nx(i, j) + q1(i, j, 2) * ny(i, j) +
                           q1(i, j, 3) * nz(i, j);
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx(i, j);
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny(i, j);
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz(i, j);

            // match temperature
            q0(i, j, 4) = q1(i, j, 4);
            // match species
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }
    eos(b, th, face._nface, "prims");
  } else if (terms.compare("postDqDxyz") == 0) {

    // Only applied to first halo slice
    threeDsubview dqdx0 = getFaceSlice(b.dqdx, face._nface, firstHaloIdx);
    threeDsubview dqdy0 = getFaceSlice(b.dqdy, face._nface, firstHaloIdx);
    threeDsubview dqdz0 = getFaceSlice(b.dqdz, face._nface, firstHaloIdx);

    threeDsubview dqdx1 =
        getFaceSlice(b.dqdx, face._nface, firstInteriorCellIdx);
    threeDsubview dqdy1 =
        getFaceSlice(b.dqdy, face._nface, firstInteriorCellIdx);
    threeDsubview dqdz1 =
        getFaceSlice(b.dqdz, face._nface, firstInteriorCellIdx);

    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(dqdx1.extent(0)),
                             static_cast<long>(dqdx1.extent(1)), b.ne});
    Kokkos::parallel_for(
        "Adia slip visc terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j, const int l) {
          // negate all gradients
          dqdx0(i, j, l) = -dqdx1(i, j, l);
          dqdy0(i, j, l) = -dqdy1(i, j, l);
          dqdz0(i, j, l) = -dqdz1(i, j, l);
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
                   b.nj, b.nk, ng, face._nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face._nface, firstInteriorCellIdx);
    twoDsubview nx, ny, nz;
    switch (face._nface) {
    case 1:
    case 2:
      nx = getFaceSlice(b.inx, face._nface, blockFaceIdx);
      ny = getFaceSlice(b.iny, face._nface, blockFaceIdx);
      nz = getFaceSlice(b.inz, face._nface, blockFaceIdx);
      break;
    case 3:
    case 4:
      nx = getFaceSlice(b.jnx, face._nface, blockFaceIdx);
      ny = getFaceSlice(b.jny, face._nface, blockFaceIdx);
      nz = getFaceSlice(b.jnz, face._nface, blockFaceIdx);
      break;
    case 5:
    case 6:
      nx = getFaceSlice(b.knx, face._nface, blockFaceIdx);
      ny = getFaceSlice(b.kny, face._nface, blockFaceIdx);
      nz = getFaceSlice(b.knz, face._nface, blockFaceIdx);
      break;
    }

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face._nface, firstHaloIdx);

      Kokkos::parallel_for(
          "Adia moving wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn = q1(i, j, 1) * nx(i, j) + q1(i, j, 2) * ny(i, j) +
                           q1(i, j, 3) * nz(i, j);
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx(i, j);
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny(i, j);
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz(i, j);

            // match temperature
            q0(i, j, 4) = q1(i, j, 4);
            // match species
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }
    eos(b, th, face._nface, "prims");

  } else if (terms.compare("preDqDxyz") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face._nface, firstInteriorCellIdx);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face._nface, firstHaloIdx);
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

    threeDsubview dqdx0 = getFaceSlice(b.dqdx, face._nface, firstHaloIdx);
    threeDsubview dqdy0 = getFaceSlice(b.dqdy, face._nface, firstHaloIdx);
    threeDsubview dqdz0 = getFaceSlice(b.dqdz, face._nface, firstHaloIdx);

    threeDsubview dqdx1 =
        getFaceSlice(b.dqdx, face._nface, firstInteriorCellIdx);
    threeDsubview dqdy1 =
        getFaceSlice(b.dqdy, face._nface, firstInteriorCellIdx);
    threeDsubview dqdz1 =
        getFaceSlice(b.dqdz, face._nface, firstInteriorCellIdx);

    MDRange2 range_face = MDRange2({0, 0}, {dqdx1.extent(0), dqdx1.extent(1)});
    Kokkos::parallel_for(
        "Adia moving wall postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          // negate pressure,  neumann velocity gradients
          dqdx0(i, j, 0) = -dqdx1(i, j, 0);
          dqdx0(i, j, 1) = dqdx1(i, j, 1);
          dqdx0(i, j, 2) = dqdx1(i, j, 2);
          dqdx0(i, j, 3) = dqdx1(i, j, 3);

          dqdy0(i, j, 0) = -dqdy1(i, j, 0);
          dqdy0(i, j, 1) = dqdy1(i, j, 1);
          dqdy0(i, j, 2) = dqdy1(i, j, 2);
          dqdy0(i, j, 3) = dqdy1(i, j, 3);

          dqdz0(i, j, 0) = -dqdz1(i, j, 0);
          dqdz0(i, j, 1) = dqdz1(i, j, 1);
          dqdz0(i, j, 2) = dqdz1(i, j, 2);
          dqdz0(i, j, 3) = dqdz1(i, j, 3);

          // negate temp and species gradient (so gradient evaluates to zero
          // on wall)
          dqdx0(i, j, 4) = -dqdx1(i, j, 4);
          dqdy0(i, j, 4) = -dqdy1(i, j, 4);
          dqdz0(i, j, 4) = -dqdz1(i, j, 4);

          for (int n = 5; n < b.ne; n++) {
            dqdx0(i, j, n) = -dqdx1(i, j, n);
            dqdy0(i, j, n) = -dqdy1(i, j, n);
            dqdz0(i, j, n) = -dqdz1(i, j, n);
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
                   b.nj, b.nk, ng, face._nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face._nface, firstInteriorCellIdx);
    twoDsubview nx, ny, nz;
    switch (face._nface) {
    case 1:
    case 2:
      nx = getFaceSlice(b.inx, face._nface, blockFaceIdx);
      ny = getFaceSlice(b.iny, face._nface, blockFaceIdx);
      nz = getFaceSlice(b.inz, face._nface, blockFaceIdx);
      break;
    case 3:
    case 4:
      nx = getFaceSlice(b.jnx, face._nface, blockFaceIdx);
      ny = getFaceSlice(b.jny, face._nface, blockFaceIdx);
      nz = getFaceSlice(b.jnz, face._nface, blockFaceIdx);
      break;
    case 5:
    case 6:
      nx = getFaceSlice(b.knx, face._nface, blockFaceIdx);
      ny = getFaceSlice(b.kny, face._nface, blockFaceIdx);
      nz = getFaceSlice(b.knz, face._nface, blockFaceIdx);
      break;
    }

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face._nface, firstHaloIdx);

      Kokkos::parallel_for(
          "isoT no slip wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn = q1(i, j, 1) * nx(i, j) + q1(i, j, 2) * ny(i, j) +
                           q1(i, j, 3) * nz(i, j);
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx(i, j);
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny(i, j);
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz(i, j);

            // set temperature
            q0(i, j, 4) = face.qBcVals(i, j, 4);
            // match species
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }
    eos(b, th, face._nface, "prims");

  } else if (terms.compare("preDqDxyz") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face._nface, firstInteriorCellIdx);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face._nface, firstHaloIdx);

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

    threeDsubview dqdx0 = getFaceSlice(b.dqdx, face._nface, firstHaloIdx);
    threeDsubview dqdy0 = getFaceSlice(b.dqdy, face._nface, firstHaloIdx);
    threeDsubview dqdz0 = getFaceSlice(b.dqdz, face._nface, firstHaloIdx);

    threeDsubview dqdx1 =
        getFaceSlice(b.dqdx, face._nface, firstInteriorCellIdx);
    threeDsubview dqdy1 =
        getFaceSlice(b.dqdy, face._nface, firstInteriorCellIdx);
    threeDsubview dqdz1 =
        getFaceSlice(b.dqdz, face._nface, firstInteriorCellIdx);

    MDRange2 range_face = MDRange2({0, 0}, {dqdx1.extent(0), dqdx1.extent(1)});
    Kokkos::parallel_for(
        "isoT no slip postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          // negate pressure,  neumann velocity,temperature gradients
          dqdx0(i, j, 0) = -dqdx1(i, j, 0);
          dqdx0(i, j, 1) = dqdx1(i, j, 1);
          dqdx0(i, j, 2) = dqdx1(i, j, 2);
          dqdx0(i, j, 3) = dqdx1(i, j, 3);
          dqdx0(i, j, 4) = dqdx1(i, j, 4);

          dqdy0(i, j, 0) = -dqdy1(i, j, 0);
          dqdy0(i, j, 1) = dqdy1(i, j, 1);
          dqdy0(i, j, 2) = dqdy1(i, j, 2);
          dqdy0(i, j, 3) = dqdy1(i, j, 3);
          dqdy0(i, j, 4) = dqdy1(i, j, 4);

          dqdz0(i, j, 0) = -dqdz1(i, j, 0);
          dqdz0(i, j, 1) = dqdz1(i, j, 1);
          dqdz0(i, j, 2) = dqdz1(i, j, 2);
          dqdz0(i, j, 3) = dqdz1(i, j, 3);
          dqdz0(i, j, 4) = dqdz1(i, j, 4);

          // negatespecies gradient (so gradient evaluates to zero
          // on wall)
          for (int n = 5; n < b.ne; n++) {
            dqdx0(i, j, n) = -dqdx1(i, j, n);
            dqdy0(i, j, n) = -dqdy1(i, j, n);
            dqdz0(i, j, n) = -dqdz1(i, j, n);
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
                   b.nj, b.nk, ng, face._nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face._nface, firstInteriorCellIdx);
    twoDsubview nx, ny, nz;
    switch (face._nface) {
    case 1:
    case 2:
      nx = getFaceSlice(b.inx, face._nface, blockFaceIdx);
      ny = getFaceSlice(b.iny, face._nface, blockFaceIdx);
      nz = getFaceSlice(b.inz, face._nface, blockFaceIdx);
      break;
    case 3:
    case 4:
      nx = getFaceSlice(b.jnx, face._nface, blockFaceIdx);
      ny = getFaceSlice(b.jny, face._nface, blockFaceIdx);
      nz = getFaceSlice(b.jnz, face._nface, blockFaceIdx);
      break;
    case 5:
    case 6:
      nx = getFaceSlice(b.knx, face._nface, blockFaceIdx);
      ny = getFaceSlice(b.kny, face._nface, blockFaceIdx);
      nz = getFaceSlice(b.knz, face._nface, blockFaceIdx);
      break;
    }

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face._nface, firstHaloIdx);

      Kokkos::parallel_for(
          "isoT slip wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // flip velo on wall
            double uDotn = q1(i, j, 1) * nx(i, j) + q1(i, j, 2) * ny(i, j) +
                           q1(i, j, 3) * nz(i, j);
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx(i, j);
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny(i, j);
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz(i, j);

            // set temperature
            q0(i, j, 4) = face.qBcVals(i, j, 4);
            // match species
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }
    eos(b, th, face._nface, "prims");

  } else if (terms.compare("postDqDxyz") == 0) {

    threeDsubview dqdx0 = getFaceSlice(b.dqdx, face._nface, firstHaloIdx);
    threeDsubview dqdy0 = getFaceSlice(b.dqdy, face._nface, firstHaloIdx);
    threeDsubview dqdz0 = getFaceSlice(b.dqdz, face._nface, firstHaloIdx);

    threeDsubview dqdx1 =
        getFaceSlice(b.dqdx, face._nface, firstInteriorCellIdx);
    threeDsubview dqdy1 =
        getFaceSlice(b.dqdy, face._nface, firstInteriorCellIdx);
    threeDsubview dqdz1 =
        getFaceSlice(b.dqdz, face._nface, firstInteriorCellIdx);

    MDRange2 range_face = MDRange2({0, 0}, {dqdx1.extent(0), dqdx1.extent(1)});
    Kokkos::parallel_for(
        "isoT slip visc terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          // negate velocity gradients
          dqdx0(i, j, 0) = -dqdx1(i, j, 0);
          dqdx0(i, j, 1) = -dqdx1(i, j, 1);
          dqdx0(i, j, 2) = -dqdx1(i, j, 2);
          dqdx0(i, j, 3) = -dqdx1(i, j, 3);

          dqdy0(i, j, 0) = -dqdy1(i, j, 0);
          dqdy0(i, j, 1) = -dqdy1(i, j, 1);
          dqdy0(i, j, 2) = -dqdy1(i, j, 2);
          dqdy0(i, j, 3) = -dqdy1(i, j, 3);

          dqdz0(i, j, 0) = -dqdz1(i, j, 0);
          dqdz0(i, j, 1) = -dqdz1(i, j, 1);
          dqdz0(i, j, 2) = -dqdz1(i, j, 2);
          dqdz0(i, j, 3) = -dqdz1(i, j, 3);

          // neumann temp gradients
          dqdx0(i, j, 4) = dqdx1(i, j, 4);
          dqdy0(i, j, 4) = dqdy1(i, j, 4);
          dqdz0(i, j, 4) = dqdz1(i, j, 4);

          // negate species gradient (so gradient evaluates to zero
          // on wall)
          for (int n = 5; n < b.ne; n++) {
            dqdx0(i, j, n) = -dqdx1(i, j, n);
            dqdy0(i, j, n) = -dqdy1(i, j, n);
            dqdz0(i, j, n) = -dqdz1(i, j, n);
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
                   b.nj, b.nk, ng, face._nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face._nface, firstInteriorCellIdx);
    twoDsubview nx, ny, nz;
    switch (face._nface) {
    case 1:
    case 2:
      nx = getFaceSlice(b.inx, face._nface, blockFaceIdx);
      ny = getFaceSlice(b.iny, face._nface, blockFaceIdx);
      nz = getFaceSlice(b.inz, face._nface, blockFaceIdx);
      break;
    case 3:
    case 4:
      nx = getFaceSlice(b.jnx, face._nface, blockFaceIdx);
      ny = getFaceSlice(b.jny, face._nface, blockFaceIdx);
      nz = getFaceSlice(b.jnz, face._nface, blockFaceIdx);
      break;
    case 5:
    case 6:
      nx = getFaceSlice(b.knx, face._nface, blockFaceIdx);
      ny = getFaceSlice(b.kny, face._nface, blockFaceIdx);
      nz = getFaceSlice(b.knz, face._nface, blockFaceIdx);
      break;
    }

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face._nface, firstHaloIdx);

      Kokkos::parallel_for(
          "Iso T moving wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn = q1(i, j, 1) * nx(i, j) + q1(i, j, 2) * ny(i, j) +
                           q1(i, j, 3) * nz(i, j);
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx(i, j);
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny(i, j);
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz(i, j);

            // set temperature
            q0(i, j, 4) = face.qBcVals(i, j, 4);
            // match species
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }
    eos(b, th, face._nface, "prims");

  } else if (terms.compare("preDqDxyz") == 0) {

    threeDsubview q1 = getFaceSlice(b.q, face._nface, firstInteriorCellIdx);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face._nface, firstHaloIdx);

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

    threeDsubview dqdx0 = getFaceSlice(b.dqdx, face._nface, firstHaloIdx);
    threeDsubview dqdy0 = getFaceSlice(b.dqdy, face._nface, firstHaloIdx);
    threeDsubview dqdz0 = getFaceSlice(b.dqdz, face._nface, firstHaloIdx);

    threeDsubview dqdx1 =
        getFaceSlice(b.dqdx, face._nface, firstInteriorCellIdx);
    threeDsubview dqdy1 =
        getFaceSlice(b.dqdy, face._nface, firstInteriorCellIdx);
    threeDsubview dqdz1 =
        getFaceSlice(b.dqdz, face._nface, firstInteriorCellIdx);

    MDRange2 range_face = MDRange2({0, 0}, {dqdx1.extent(0), dqdx1.extent(1)});
    Kokkos::parallel_for(
        "Iso T moving wall postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j) {
          // negate pressure gradient, neumann velocity, temperature gradients
          dqdx0(i, j, 0) = -dqdx1(i, j, 0);
          dqdx0(i, j, 1) = dqdx1(i, j, 1);
          dqdx0(i, j, 2) = dqdx1(i, j, 2);
          dqdx0(i, j, 3) = dqdx1(i, j, 3);
          dqdx0(i, j, 4) = dqdx1(i, j, 4);

          dqdy0(i, j, 0) = -dqdy1(i, j, 0);
          dqdy0(i, j, 1) = dqdy1(i, j, 1);
          dqdy0(i, j, 2) = dqdy1(i, j, 2);
          dqdy0(i, j, 3) = dqdy1(i, j, 3);
          dqdy0(i, j, 4) = dqdy1(i, j, 4);

          dqdz0(i, j, 0) = -dqdz1(i, j, 0);
          dqdz0(i, j, 1) = dqdz1(i, j, 1);
          dqdz0(i, j, 2) = dqdz1(i, j, 2);
          dqdz0(i, j, 3) = dqdz1(i, j, 3);
          dqdz0(i, j, 4) = dqdz1(i, j, 4);

          // negate species gradient (so gradient evaluates to zero on wall)
          for (int n = 5; n < b.ne; n++) {
            dqdx0(i, j, n) = -dqdx1(i, j, n);
            dqdy0(i, j, n) = -dqdy1(i, j, n);
            dqdz0(i, j, n) = -dqdz1(i, j, n);
          }
        });
  }
}
