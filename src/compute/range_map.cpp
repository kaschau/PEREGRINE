#include "block_.hpp"
#include "kokkos_types.hpp"
#include <stdexcept>

MDRange3 get_range3(block_ b, const int nface, const int indxI /*=0*/,
                    const int indxJ /*=0*/, const int indxK /*=0*/) {

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
    throw std::invalid_argument("Unknown argument to get_range3");
  }

  return range;
}

threeDsubview getHaloSlice(fourDview view, const int nface, int slice) {

  threeDsubview subview;
  switch (nface) {
  case 1:
  case 2:
    // face 1 halo
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
    std::cout << nface;
    throw std::invalid_argument(" <-- Unknown argument to getHaloSlice");
  }

  return subview;
}

twoDsubview getHaloSlice(threeDview view, const int nface, int slice) {

  twoDsubview subview;
  switch (nface) {
  case 1:
  case 2:
    // face 1 halo
    subview =
        Kokkos::subview(view, slice, Kokkos::ALL, Kokkos::ALL);
    break;
  case 3:
  case 4:
    // face 3,4 face slices
    subview =
        Kokkos::subview(view, Kokkos::ALL, slice, Kokkos::ALL);
    break;
  case 5:
  case 6:
    // face 5,6 face slices
    subview =
        Kokkos::subview(view, Kokkos::ALL, Kokkos::ALL, slice);
    break;
  default:
    std::cout << nface;
    throw std::invalid_argument(" <-- Unknown argument to getHaloSlice");
  }

  return subview;
}

void setHaloSlices(int &s0, int &s1, int &s2, int &plus, const int ni,
                   const int nj, const int nk, const int ng, const int nface) {
  switch (nface) {
  case 1:
  case 3:
  case 5:
    s0 = ng - 1;
    s1 = ng;
    s2 = ng + 1;
    plus = 1;
    break;
  case 2:
    s0 = ni + ng - 1;
    s1 = ni + ng - 2;
    s2 = ni + ng - 3;
    plus = -1;
    break;
  case 4:
    s0 = nj + ng - 1;
    s1 = nj + ng - 2;
    s2 = nj + ng - 3;
    plus = -1;
    break;
  case 6:
    s0 = nk + ng - 1;
    s1 = nk + ng - 2;
    s2 = nk + ng - 3;
    plus = -1;
    break;
  default:
    throw std::invalid_argument("Unknown argument setHaloSlice");
  }
}
