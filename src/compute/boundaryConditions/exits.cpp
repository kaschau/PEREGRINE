#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include "face_.hpp"
#include "kokkos_types.hpp"
#include "thtrdat_.hpp"

void constantPressureSubsonicExit(
    block_ b, face_ face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ th, const std::string terms, const double /*tme*/) {
  //-------------------------------------------------------------------------------------------|
  // Apply BC to face, slice by slice.
  //-------------------------------------------------------------------------------------------|
  const int ng = b.ng;
  int s0, s1, s2, plus;
  setHaloSlices(s0, s1, s2, plus, b.ni, b.nj, b.nk, ng, face._nface);

  if (terms.compare("euler") == 0) {

    threeDsubview q1 = getHaloSlice(b.q, face._nface, s1);
    twoDsubview nx, ny, nz;

    if (face._nface == 1 || face._nface == 2) {
      nx = getHaloSlice(b.inx, face._nface, s1);
      ny = getHaloSlice(b.iny, face._nface, s1);
      nz = getHaloSlice(b.inz, face._nface, s1);
    } else if (face._nface == 3 || face._nface == 4) {
      nx = getHaloSlice(b.jnx, face._nface, s1);
      ny = getHaloSlice(b.jny, face._nface, s1);
      nz = getHaloSlice(b.jnz, face._nface, s1);
    } else if (face._nface == 5 || face._nface == 6) {
      nx = getHaloSlice(b.knx, face._nface, s1);
      ny = getHaloSlice(b.kny, face._nface, s1);
      nz = getHaloSlice(b.knz, face._nface, s1);
    }

    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    double dplus = -plus; // need outward normal
    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;
      s2 += plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);
      threeDsubview q2 = getHaloSlice(b.q, face._nface, s2);

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

            // extrapolate everything else
            for (int l = 4; l < b.ne; l++) {
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
          "Constant pressure subsonic exit viscous terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j, const int l) {
            // neumann all gradients
            dqdx0(i, j, l) = dqdx1(i, j, l);
            dqdy0(i, j, l) = dqdy1(i, j, l);
            dqdz0(i, j, l) = dqdz1(i, j, l);
          });
    }
  } else if (terms.compare("strict") == 0) {
  }
}

void supersonicExit(
    block_ b, face_ face,
    const std::function<void(block_, thtrdat_, int, std::string)> &eos,
    const thtrdat_ th, const std::string terms, const double /*tme*/) {
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
          "Supersonic exit euler terms", range_face,
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
          "Supersonic exit viscous terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j, const int l) {
            // neumann all gradients
            dqdx0(i, j, l) = dqdx1(i, j, l);
            dqdy0(i, j, l) = dqdy1(i, j, l);
            dqdz0(i, j, l) = dqdz1(i, j, l);
          });
    }
  } else if (terms.compare("strict") == 0) {
  }
}
