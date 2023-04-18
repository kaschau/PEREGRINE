#include "block_.hpp"
#include "kokkosTypes.hpp"
#include <stdexcept>

MDRange3 getRange3(const block_ &b, const int &nface, const int &indxI /*=0*/,
                   const int &indxJ /*=0*/, const int &indxK /*=0*/) {

  MDRange3 range;

  switch (nface) {
  case -1:
    // total block
    range = MDRange3({0, 0, 0}, {b.ni + 2 * b.ng - 1, b.nj + 2 * b.ng - 1,
                                 b.nk + 2 * b.ng - 1});
    break;
  case 0:
    // interior
    range = MDRange3({b.ng, b.ng, b.ng},
                     {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});
    break;
  case 1:
    // face 1 halo
    range =
        MDRange3({0, 0, 0}, {b.ng, b.nj + 2 * b.ng - 1, b.nk + 2 * b.ng - 1});
    break;
  case 2:
    // face 2 halo
    range = MDRange3(
        {b.ni + b.ng - 1, 0, 0},
        {b.ni + 2 * b.ng - 1, b.nj + 2 * b.ng - 1, b.nk + 2 * b.ng - 1});
    break;
  case 3:
    // face 3 halo
    range =
        MDRange3({0, 0, 0}, {b.ni + 2 * b.ng - 1, b.ng, b.nk + 2 * b.ng - 1});
    break;
  case 4:
    // face 4 halo
    range = MDRange3(
        {0, b.nj + b.ng - 1, 0},
        {b.ni + 2 * b.ng - 1, b.nj + 2 * b.ng - 1, b.nk + 2 * b.ng - 1});
    break;
  case 5:
    // face 5 halo
    range =
        MDRange3({0, 0, 0}, {b.ni + 2 * b.ng - 1, b.nj + 2 * b.ng - 1, b.ng});
    break;
  case 6:
    // face 6 halo
    range = MDRange3(
        {0, 0, b.nk + b.ng - 1},
        {b.ni + 2 * b.ng - 1, b.nj + 2 * b.ng - 1, b.nk + 2 * b.ng - 1});
    break;
  case 10:
    // specify i,j,k turn it into a function call (kinda)
    range = MDRange3({indxI, indxJ, indxK}, {indxI + 1, indxJ + 1, indxK + 1});
    break;
  default:
    throw std::invalid_argument("Unknown argument to getRange3");
  }

  return range;
}

twoDsubview getFaceSlice(const threeDview &view, const int &nface,
                         const int &slice) {

  twoDsubview subview;
  switch (nface) {
  case 1:
  case 2:
    // face 1,2 halo
    subview = Kokkos::subview(view, slice, Kokkos::ALL, Kokkos::ALL);
    break;
  case 3:
  case 4:
    // face 3,4 face slices
    subview = Kokkos::subview(view, Kokkos::ALL, slice, Kokkos::ALL);
    break;
  case 5:
  case 6:
    // face 5,6 face slices
    subview = Kokkos::subview(view, Kokkos::ALL, Kokkos::ALL, slice);
    break;
  default:
    throw std::invalid_argument(" <-- Unknown argument to getFaceSlice");
  }

  return subview;
}

threeDsubview getFaceSlice(const fourDview &view, const int &nface,
                           const int &slice) {

  threeDsubview subview;
  switch (nface) {
  case 1:
  case 2:
    // face 1,2 halo
    subview =
        Kokkos::subview(view, slice, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    break;
  case 3:
  case 4:
    // face 3,4 face slices
    subview =
        Kokkos::subview(view, Kokkos::ALL, slice, Kokkos::ALL, Kokkos::ALL);
    break;
  case 5:
  case 6:
    // face 5,6 face slices
    subview =
        Kokkos::subview(view, Kokkos::ALL, Kokkos::ALL, slice, Kokkos::ALL);
    break;
  default:
    throw std::invalid_argument(" <-- Unknown argument to getFaceSlice");
  }

  return subview;
}

void getFaceSliceIdxs(int &firstHaloIdx, int &firstInteriorCellIdx,
                      int &blockFaceIdx, int &plus, const int &ni,
                      const int &nj, const int &nk, const int &ng,
                      const int &nface) {
  // For low faces (1,3,5) plus = 1, i.e. inward normal for the face. For high
  // faces (2,4,6) plus = -1, i.e. inward normal for the face.
  switch (nface) {
  case 1:
  case 3:
  case 5:
    firstHaloIdx = ng - 1;
    firstInteriorCellIdx = ng;
    blockFaceIdx = ng;
    plus = 1;
    break;
  case 2:
    firstHaloIdx = ni + ng - 1;
    firstInteriorCellIdx = ni + ng - 2;
    blockFaceIdx = ni + ng - 1;
    plus = -1;
    break;
  case 4:
    firstHaloIdx = nj + ng - 1;
    firstInteriorCellIdx = nj + ng - 2;
    blockFaceIdx = nj + ng - 1;
    plus = -1;
    break;
  case 6:
    firstHaloIdx = nk + ng - 1;
    firstInteriorCellIdx = nk + ng - 2;
    blockFaceIdx = nk + ng - 1;
    plus = -1;
    break;
  default:
    throw std::invalid_argument("Unknown argument setHaloSlice");
  }
}
