#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include "face_.hpp"
#include "kokkos_types.hpp"
#include "thtrdat_.hpp"

void adiabaticNoSlipWall(
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
            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // flip velo on wall
            q0(i, j, 1) = -q1(i, j, 1);
            q0(i, j, 2) = -q1(i, j, 2);
            q0(i, j, 3) = -q1(i, j, 3);

            // match temperature
            q0(i, j, 4) = q1(i, j, 4);
            // match species
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }
    eos(b, th, face._nface, "prims");

  } else if (terms.compare("viscous") == 0) {

    threeDsubview dqdx1 = getHaloSlice(b.dqdx, face._nface, s1);
    threeDsubview dqdy1 = getHaloSlice(b.dqdy, face._nface, s1);
    threeDsubview dqdz1 = getHaloSlice(b.dqdz, face._nface, s1);
    MDRange2 range_face = MDRange2({0, 0}, {dqdx1.extent(0), dqdx1.extent(1)});

    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;
      s2 += plus * g;

      threeDsubview dqdx0 = getHaloSlice(b.dqdx, face._nface, s0);
      threeDsubview dqdy0 = getHaloSlice(b.dqdy, face._nface, s0);
      threeDsubview dqdz0 = getHaloSlice(b.dqdz, face._nface, s0);

      threeDsubview dqdx2 = getHaloSlice(b.dqdx, face._nface, s2);
      threeDsubview dqdy2 = getHaloSlice(b.dqdy, face._nface, s2);
      threeDsubview dqdz2 = getHaloSlice(b.dqdz, face._nface, s2);

      Kokkos::parallel_for(
          "Constant velocity subsonic inlet euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // extrapolate pressure, velocity gradients
            dqdx0(i, j, 0) = -dqdx1(i, j, 0);
            dqdx0(i, j, 1) = 2.0 * dqdx1(i, j, 1) - dqdx2(i, j, 1);
            dqdx0(i, j, 2) = 2.0 * dqdx1(i, j, 2) - dqdx2(i, j, 2);
            dqdx0(i, j, 3) = 2.0 * dqdx1(i, j, 3) - dqdx2(i, j, 3);

            dqdy0(i, j, 0) = -dqdy1(i, j, 0);
            dqdy0(i, j, 1) = 2.0 * dqdy1(i, j, 1) - dqdy2(i, j, 1);
            dqdy0(i, j, 2) = 2.0 * dqdy1(i, j, 2) - dqdy2(i, j, 2);
            dqdy0(i, j, 3) = 2.0 * dqdy1(i, j, 3) - dqdy2(i, j, 3);

            dqdz0(i, j, 0) = -dqdz1(i, j, 0);
            dqdz0(i, j, 1) = 2.0 * dqdz1(i, j, 1) - dqdz2(i, j, 1);
            dqdz0(i, j, 2) = 2.0 * dqdz1(i, j, 2) - dqdz2(i, j, 2);
            dqdz0(i, j, 3) = 2.0 * dqdz1(i, j, 3) - dqdz2(i, j, 3);

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
}

void adiabaticSlipWall(
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

    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);

      double dplus = static_cast<double>(plus);
      Kokkos::parallel_for(
          "Constant velocity subsonic inlet euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // flip velo on wall
            double uDotn = q1(i, j, 1) * nx(i, j) + q1(i, j, 2) * ny(i, j) +
                           q1(i, j, 3) * nz(i, j);
            q0(i, j, 1) = q1(i, j, 1) - dplus * 2.0 * uDotn * nx(i, j);
            q0(i, j, 2) = q1(i, j, 2) - dplus * 2.0 * uDotn * ny(i, j);
            q0(i, j, 3) = q1(i, j, 3) - dplus * 2.0 * uDotn * nz(i, j);

            // match temperature
            q0(i, j, 4) = q1(i, j, 4);
            // match species
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }
    eos(b, th, face._nface, "prims");

  } else if (terms.compare("viscous") == 0) {

    threeDsubview dqdx1 = getHaloSlice(b.dqdx, face._nface, s1);
    threeDsubview dqdy1 = getHaloSlice(b.dqdy, face._nface, s1);
    threeDsubview dqdz1 = getHaloSlice(b.dqdz, face._nface, s1);
    MDRange2 range_face = MDRange2({0, 0}, {dqdx1.extent(0), dqdx1.extent(1)});

    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;
      threeDsubview dqdx0 = getHaloSlice(b.dqdx, face._nface, s0);
      threeDsubview dqdy0 = getHaloSlice(b.dqdy, face._nface, s0);
      threeDsubview dqdz0 = getHaloSlice(b.dqdz, face._nface, s0);

      Kokkos::parallel_for(
          "Constant velocity subsonic inlet euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // neumann velocity gradients
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
}

void adiabaticMovingWall(
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

      Kokkos::parallel_for(
          "Constant velocity subsonic inlet euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // flip velo on wall
            q0(i, j, 1) = 2.0 * face.qBcVals(i, j, 1) - q1(i, j, 1);
            q0(i, j, 2) = 2.0 * face.qBcVals(i, j, 2) - q1(i, j, 2);
            q0(i, j, 3) = 2.0 * face.qBcVals(i, j, 3) - q1(i, j, 3);

            // match temperature
            q0(i, j, 4) = q1(i, j, 4);
            // match species
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }
    eos(b, th, face._nface, "prims");

  } else if (terms.compare("viscous") == 0) {

    threeDsubview dqdx1 = getHaloSlice(b.dqdx, face._nface, s1);
    threeDsubview dqdy1 = getHaloSlice(b.dqdy, face._nface, s1);
    threeDsubview dqdz1 = getHaloSlice(b.dqdz, face._nface, s1);

    MDRange2 range_face = MDRange2({0, 0}, {dqdx1.extent(0), dqdx1.extent(1)});

    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;
      s2 += plus * g;
      threeDsubview dqdx0 = getHaloSlice(b.dqdx, face._nface, s0);
      threeDsubview dqdy0 = getHaloSlice(b.dqdy, face._nface, s0);
      threeDsubview dqdz0 = getHaloSlice(b.dqdz, face._nface, s0);

      threeDsubview dqdx2 = getHaloSlice(b.dqdx, face._nface, s2);
      threeDsubview dqdy2 = getHaloSlice(b.dqdy, face._nface, s2);
      threeDsubview dqdz2 = getHaloSlice(b.dqdz, face._nface, s2);

      Kokkos::parallel_for(
          "Constant velocity subsonic inlet euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // extrapolate velocity gradients
            dqdx0(i, j, 0) = 2.0 * dqdx1(i, j, 0) - dqdx2(i, j, 0);
            dqdx0(i, j, 1) = 2.0 * dqdx1(i, j, 1) - dqdx2(i, j, 1);
            dqdx0(i, j, 2) = 2.0 * dqdx1(i, j, 2) - dqdx2(i, j, 2);
            dqdx0(i, j, 3) = 2.0 * dqdx1(i, j, 3) - dqdx2(i, j, 3);

            dqdy0(i, j, 0) = 2.0 * dqdy1(i, j, 0) - dqdy2(i, j, 0);
            dqdy0(i, j, 1) = 2.0 * dqdy1(i, j, 1) - dqdy2(i, j, 1);
            dqdy0(i, j, 2) = 2.0 * dqdy1(i, j, 2) - dqdy2(i, j, 2);
            dqdy0(i, j, 3) = 2.0 * dqdy1(i, j, 3) - dqdy2(i, j, 3);

            dqdz0(i, j, 0) = 2.0 * dqdz1(i, j, 0) - dqdz2(i, j, 0);
            dqdz0(i, j, 1) = 2.0 * dqdz1(i, j, 1) - dqdz2(i, j, 1);
            dqdz0(i, j, 2) = 2.0 * dqdz1(i, j, 2) - dqdz2(i, j, 2);
            dqdz0(i, j, 3) = 2.0 * dqdz1(i, j, 3) - dqdz2(i, j, 3);

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
}

void isoTMovingWall(
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

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);

      Kokkos::parallel_for(
          "Constant velocity subsonic inlet euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // flip velo on wall
            q0(i, j, 1) = 2.0 * face.qBcVals(i, j, 1) - q1(i, j, 1);
            q0(i, j, 2) = 2.0 * face.qBcVals(i, j, 2) - q1(i, j, 2);
            q0(i, j, 3) = 2.0 * face.qBcVals(i, j, 3) - q1(i, j, 3);

            // set temperature
            q0(i, j, 4) = 2.0 * face.qBcVals(i, j, 4) - q1(i, j, 4);
            // match species
            for (int n = 5; n < b.ne; n++) {
              q0(i, j, n) = q1(i, j, n);
            }
          });
    }
    eos(b, th, face._nface, "prims");

  } else if (terms.compare("viscous") == 0) {

    threeDsubview dqdx1 = getHaloSlice(b.dqdx, face._nface, s1);
    threeDsubview dqdy1 = getHaloSlice(b.dqdy, face._nface, s1);
    threeDsubview dqdz1 = getHaloSlice(b.dqdz, face._nface, s1);

    MDRange2 range_face = MDRange2({0, 0}, {dqdx1.extent(0), dqdx1.extent(1)});

    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;
      s2 += plus * g;
      threeDsubview dqdx0 = getHaloSlice(b.dqdx, face._nface, s0);
      threeDsubview dqdy0 = getHaloSlice(b.dqdy, face._nface, s0);
      threeDsubview dqdz0 = getHaloSlice(b.dqdz, face._nface, s0);

      threeDsubview dqdx2 = getHaloSlice(b.dqdx, face._nface, s2);
      threeDsubview dqdy2 = getHaloSlice(b.dqdy, face._nface, s2);
      threeDsubview dqdz2 = getHaloSlice(b.dqdz, face._nface, s2);

      Kokkos::parallel_for(
          "Constant velocity subsonic inlet euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // extrapolate velocity gradients
            // extrapolate temp gradient
            dqdx0(i, j, 0) = 2.0 * dqdx1(i, j, 0) - dqdx2(i, j, 0);
            dqdx0(i, j, 1) = 2.0 * dqdx1(i, j, 1) - dqdx2(i, j, 1);
            dqdx0(i, j, 2) = 2.0 * dqdx1(i, j, 2) - dqdx2(i, j, 2);
            dqdx0(i, j, 3) = 2.0 * dqdx1(i, j, 3) - dqdx2(i, j, 3);
            dqdx0(i, j, 4) = 2.0 * dqdx1(i, j, 4) - dqdx2(i, j, 4);

            dqdy0(i, j, 0) = 2.0 * dqdy1(i, j, 0) - dqdy2(i, j, 0);
            dqdy0(i, j, 1) = 2.0 * dqdy1(i, j, 1) - dqdy2(i, j, 1);
            dqdy0(i, j, 2) = 2.0 * dqdy1(i, j, 2) - dqdy2(i, j, 2);
            dqdy0(i, j, 3) = 2.0 * dqdy1(i, j, 3) - dqdy2(i, j, 3);
            dqdy0(i, j, 4) = 2.0 * dqdy1(i, j, 4) - dqdy2(i, j, 4);

            dqdz0(i, j, 0) = 2.0 * dqdz1(i, j, 0) - dqdz2(i, j, 0);
            dqdz0(i, j, 1) = 2.0 * dqdz1(i, j, 1) - dqdz2(i, j, 1);
            dqdz0(i, j, 2) = 2.0 * dqdz1(i, j, 2) - dqdz2(i, j, 2);
            dqdz0(i, j, 3) = 2.0 * dqdz1(i, j, 3) - dqdz2(i, j, 3);
            dqdz0(i, j, 4) = 2.0 * dqdz1(i, j, 4) - dqdz2(i, j, 4);

            // negate species gradient (so gradient evaluates to zero on wall)
            for (int n = 5; n < b.ne; n++) {
              dqdx0(i, j, n) = -dqdx1(i, j, n);
              dqdy0(i, j, n) = -dqdy1(i, j, n);
              dqdz0(i, j, n) = -dqdz1(i, j, n);
            }
          });
    }
  }
}
