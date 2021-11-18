#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include "face_.hpp"
#include "kokkos_types.hpp"
#include "thtrdat_.hpp"

void constantPressureSubsonicExit(
    block_ b, const face_ face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    thtrdat_ th, std::string terms) {
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
      s2 += plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);
      threeDsubview q2 = getHaloSlice(b.q, face._nface, s2);

      Kokkos::parallel_for(
          "Constant velocity subsonic inlet euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // set pressure
            q0(i, j, 0) = 2.0 * face.qBcVals(i,j,0) - q1(i, j, 0);

            // extrapolate everything else
            for (int l = 1; l < b.ne; l++) {
              q0(i, j, l) = 2.0 * q1(i, j, l) - q2(i, j, l);
            }
          });
    }
    eos(b, th, face._nface, "prims");

  } else if (terms.compare("viscous") == 0) {

    threeDsubview dqdx1 = getHaloSlice(b.dqdx, face._nface, s1);
    threeDsubview dqdy1 = getHaloSlice(b.dqdy, face._nface, s1);
    threeDsubview dqdz1 = getHaloSlice(b.dqdz, face._nface, s1);

    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(dqdx1.extent(0)),
                             static_cast<long>(dqdx1.extent(1)), b.ne});
    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;
      threeDsubview dqdx0 = getHaloSlice(b.dqdx, face._nface, s0);
      threeDsubview dqdy0 = getHaloSlice(b.dqdy, face._nface, s0);
      threeDsubview dqdz0 = getHaloSlice(b.dqdz, face._nface, s0);

      Kokkos::parallel_for(
          "Constant velocity subsonic inlet euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j, const int l) {
            // neumann all gradients
            dqdx0(i, j, l) = dqdx1(i, j, l);
            dqdy0(i, j, l) = dqdy1(i, j, l);
            dqdz0(i, j, l) = dqdz1(i, j, l);
          });
    }
  }
}

void supersonicExit(
    block_ b, const face_ face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    thtrdat_ th, std::string terms) {
  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const int ng = b.ng;
  int s0, s1, s2, plus;
  setHaloSlices(s0, s1, s2, plus, b.ni, b.nj, b.nk, ng, face._nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getHaloSlice(b.q, face._nface, s1);
    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(q1.extent(0)),
                             static_cast<long>(q1.extent(1)), b.ne});

    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;
      s2 += plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);
      threeDsubview q2 = getHaloSlice(b.q, face._nface, s2);

      Kokkos::parallel_for(
          "Constant velocity subsonic inlet euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j, const int l) {
            // extrapolate everything
            q0(i, j, l) = 2.0 * q1(i, j, l) - q2(i, j, l);
          });
    }
    eos(b, th, face._nface, "prims");

  } else if (terms.compare("viscous") == 0) {

    threeDsubview dqdx1 = getHaloSlice(b.dqdx, face._nface, s1);
    threeDsubview dqdy1 = getHaloSlice(b.dqdy, face._nface, s1);
    threeDsubview dqdz1 = getHaloSlice(b.dqdz, face._nface, s1);

    MDRange3 range_face =
        MDRange3({0, 0, 0}, {static_cast<long>(dqdx1.extent(0)),
                             static_cast<long>(dqdx1.extent(1)), b.ne});
    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;
      threeDsubview dqdx0 = getHaloSlice(b.dqdx, face._nface, s0);
      threeDsubview dqdy0 = getHaloSlice(b.dqdy, face._nface, s0);
      threeDsubview dqdz0 = getHaloSlice(b.dqdz, face._nface, s0);

      Kokkos::parallel_for(
          "Constant velocity subsonic inlet euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j, const int l) {
            // neumann all gradients
            dqdx0(i, j, l) = dqdx1(i, j, l);
            dqdy0(i, j, l) = dqdy1(i, j, l);
            dqdz0(i, j, l) = dqdz1(i, j, l);
          });
    }
  }
}
