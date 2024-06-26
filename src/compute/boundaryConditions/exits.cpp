#include "block_.hpp"
#include "compute.hpp"
#include "face_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"
#include <Kokkos_Core.hpp>
#include <string.h>

void constantPressureSubsonicExit(
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
  int secondInteriorCellIdx = firstInteriorCellIdx + plus;

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
    double dplus = -plus; // need outward normal
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;
      threeDsubview q0 = getFaceSlice(b.q, face._nface, firstHaloIdx);
      secondInteriorCellIdx += plus * g;
      threeDsubview q2 = getFaceSlice(b.q, face._nface, secondInteriorCellIdx);

      Kokkos::parallel_for(
          "Constant pressure subsonic exit euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // set pressure
            q0(i, j, 0) = face.qBcVals(i, j, 0);

            // extrapolate velocity, unless reverse flow detected
            double uDotn = (q1(i, j, 1) * nx(i, j) + q1(i, j, 2) * ny(i, j) +
                            q1(i, j, 3) * nz(i, j)) *
                           dplus;
            if (uDotn > 0.0) {
              for (int l = 1; l <= 3; l++) {
                q0(i, j, l) = 2.0 * q1(i, j, l) - q2(i, j, l);
              }
            } else {
              // flip velocity on face (like slip wall)
              q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx(i, j) * dplus;
              q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny(i, j) * dplus;
              q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz(i, j) * dplus;
            }

            // neumann everything else
            for (int l = 4; l < b.ne; l++) {
              q0(i, j, l) = q1(i, j, l);
            }
          });
    }
    eos(b, th, face._nface, "prims");
  } else if (terms.compare("postDqDxyz") == 0) {

    // Only gets applied to first halo slice
    threeDsubview dqdx1 =
        getFaceSlice(b.dqdx, face._nface, firstInteriorCellIdx);
    threeDsubview dqdy1 =
        getFaceSlice(b.dqdy, face._nface, firstInteriorCellIdx);
    threeDsubview dqdz1 =
        getFaceSlice(b.dqdz, face._nface, firstInteriorCellIdx);

    threeDsubview dqdx0 = getFaceSlice(b.dqdx, face._nface, firstHaloIdx);
    threeDsubview dqdy0 = getFaceSlice(b.dqdy, face._nface, firstHaloIdx);
    threeDsubview dqdz0 = getFaceSlice(b.dqdz, face._nface, firstHaloIdx);

    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(dqdx1.extent(0)),
                             static_cast<long>(dqdx1.extent(1)), b.ne});
    Kokkos::parallel_for(
        "Constant pressure subsonic exit postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j, const int l) {
          // neumann all gradients
          dqdx0(i, j, l) = dqdx1(i, j, l);
          dqdy0(i, j, l) = dqdy1(i, j, l);
          dqdz0(i, j, l) = dqdz1(i, j, l);
        });
  }
}

void supersonicExit(
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
  int secondInteriorCellIdx = firstInteriorCellIdx + plus;

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
    double dplus = -plus; // need outward normal
    for (int g = 0; g < b.ng; g++) {
      firstHaloIdx -= plus * g;
      secondInteriorCellIdx += plus * g;

      threeDsubview q0 = getFaceSlice(b.q, face._nface, firstHaloIdx);
      threeDsubview q2 = getFaceSlice(b.q, face._nface, secondInteriorCellIdx);

      Kokkos::parallel_for(
          "Supersonic exit euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // extrapolate pressure (keep it positive, and wave exiting)
            q0(i, j, 0) =
                fmin(fmax(0.0, 2.0 * q1(i, j, 0) - q2(i, j, 0)), q1(i, j, 0));

            // extrapolate velocity, unless reverse flow detected
            double uDotn = (q1(i, j, 1) * nx(i, j) + q1(i, j, 2) * ny(i, j) +
                            q1(i, j, 3) * nz(i, j)) *
                           dplus;
            if (uDotn > 0.0) {
              for (int l = 1; l <= 3; l++) {
                q0(i, j, l) = 2.0 * q1(i, j, l) - q2(i, j, l);
              }
            } else {
              // flip velocity on face (like slip wall)
              q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx(i, j) * dplus;
              q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny(i, j) * dplus;
              q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz(i, j) * dplus;
            }

            // extrapolate temperature (keep it positive)
            q0(i, j, 4) = fmax(0.0, 2.0 * q1(i, j, 4) - q2(i, j, 4));
            // extrapolate species
            for (int l = 5; l < b.ne; l++) {
              q0(i, j, l) =
                  fmax(0.0, fmin(1.0, 2.0 * q1(i, j, l) - q2(i, j, l)));
            }
          });
    }
    eos(b, th, face._nface, "prims");
  } else if (terms.compare("postDqDxyz") == 0) {

    // Only applied to first halo slice
    threeDsubview dqdx1 =
        getFaceSlice(b.dqdx, face._nface, firstInteriorCellIdx);
    threeDsubview dqdy1 =
        getFaceSlice(b.dqdy, face._nface, firstInteriorCellIdx);
    threeDsubview dqdz1 =
        getFaceSlice(b.dqdz, face._nface, firstInteriorCellIdx);

    threeDsubview dqdx0 = getFaceSlice(b.dqdx, face._nface, firstHaloIdx);
    threeDsubview dqdy0 = getFaceSlice(b.dqdy, face._nface, firstHaloIdx);
    threeDsubview dqdz0 = getFaceSlice(b.dqdz, face._nface, firstHaloIdx);

    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(dqdx1.extent(0)),
                             static_cast<long>(dqdx1.extent(1)), b.ne});
    Kokkos::parallel_for(
        "Supersonic exit postDqDxyz terms", range_face,
        KOKKOS_LAMBDA(const int i, const int j, const int l) {
          // neumann all gradients
          dqdx0(i, j, l) = dqdx1(i, j, l);
          dqdy0(i, j, l) = dqdy1(i, j, l);
          dqdz0(i, j, l) = dqdz1(i, j, l);
        });
  }
}
