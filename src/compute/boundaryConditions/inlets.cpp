#include "Kokkos_Core.hpp"
#include "compute.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "face_.hpp"


Kokkos::View<double***> subviewSlice(Kokkos::View<double****> view, const int nface,
                                       int slice) {
  Kokkos::View<double***> subview;

  switch (nface) {
    case 1: case 2:
      // face 1 halo
      subview = Kokkos::subview(view, slice, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
      break;
    case 3: case 4:
      // face 3,4 face slices
      subview = Kokkos::subview(view, Kokkos::ALL, slice, Kokkos::ALL, Kokkos::ALL);
      break;
    case 5: case 6:
      // face 5,6 face slices
      subview = Kokkos::subview(view, Kokkos::ALL, Kokkos::ALL, slice, Kokkos::ALL);
      break;
    default :
      throw std::invalid_argument( "Unknown argument to subview3Slice");
  }

  return subview;
};

void getSlices(int& s0, int& s1, int& s2, int& plus, const int ni, const int nj, const int nk, const int ng, const int nface){
  switch (nface) {
    case 1: case 3: case 5:
      s0 = ng - 1;
      s1 = ng;
      s2 = ng + 1;
      plus = 1;
      break;
    case 2:
      s0 = ni + 2*ng;
      s1 = ni + 2*ng - 1;
      s2 = ni + 2*ng - 2;
      plus = -1;
      break;
    case 4:
      s0 = nj + 2*ng;
      s1 = nj + 2*ng - 1;
      s2 = nj + 2*ng - 2;
      plus = -1;
      break;
    case 6:
      s0 = nk + 2*ng;
      s1 = nk + 2*ng - 1;
      s2 = nk + 2*ng - 2;
      plus = -1;
      break;
    default:
      throw std::invalid_argument( "Unknown argument getSlice");
  }
}

void constantVelocitySubsonicInlet(block_ b, const face_ face) {
//-------------------------------------------------------------------------------------------|
// Apply BC to face, slice by slice.
//-------------------------------------------------------------------------------------------|
  const int ng = b.ng;
  int s0,s1,s2, plus;
  getSlices(s0, s1, s2, plus, b.ni, b.nj, b.nk, ng, face._nface);

  auto q1 = subviewSlice(b.q, face._nface, s1);

  for (int g=0; g < b.ng; g++) {
    s0 -= plus*g;
    s2 += plus*g;

    auto q0 = subviewSlice(b.q, face._nface, s0);
    auto q2 = subviewSlice(b.q, face._nface, s2);

    MDRange2 range_face = get_range2(b, face._nface);
    Kokkos::parallel_for("Constant velocity subsonic inlet",
                         range_face,
                         KOKKOS_LAMBDA(const int i,
                                       const int j) {

      // extrapolate pressure
      q0(i,j,0) = 2.0 * q1(i,j,0) - q2(i,j,0);

      // apply velo on face
      q0(i,j,1) = 2.0 * face.qBcVals(1) - q1(i,j,1);
      q0(i,j,2) = 2.0 * face.qBcVals(2) - q1(i,j,2);
      q0(i,j,3) = 2.0 * face.qBcVals(3) - q1(i,j,3);

      // apply temperature on face
      q0(i,j,4) = 2.0 * face.qBcVals(4) - q1(i,j,4);

      for (int n=5; n < b.ne; n++) {
        q0(i,j,n) = 2.0 * face.qBcVals(5) - q1(i,j,n);
      }

  });

  }
};
