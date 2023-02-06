#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include "face_.hpp"
#include "kokkosTypes.hpp"
#include "thtrdat_.hpp"

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
    double dplus = plus;
    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;
      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);

      Kokkos::parallel_for(
          "Adia no slip wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn = (q1(i, j, 1) * nx(i, j) + q1(i, j, 2) * ny(i, j) +
                            q1(i, j, 3) * nz(i, j)) *
                           dplus;
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx(i, j) * dplus;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny(i, j) * dplus;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz(i, j) * dplus;

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

    threeDsubview q1 = getHaloSlice(b.q, face._nface, s1);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;
      s2 += plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);
      threeDsubview q2 = getHaloSlice(b.q, face._nface, s2);

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
    threeDsubview dqdx0 = getHaloSlice(b.dqdx, face._nface, s0);
    threeDsubview dqdy0 = getHaloSlice(b.dqdy, face._nface, s0);
    threeDsubview dqdz0 = getHaloSlice(b.dqdz, face._nface, s0);

    threeDsubview dqdx1 = getHaloSlice(b.dqdx, face._nface, s1);
    threeDsubview dqdy1 = getHaloSlice(b.dqdy, face._nface, s1);
    threeDsubview dqdz1 = getHaloSlice(b.dqdz, face._nface, s1);

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
    double dplus = plus;
    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);

      Kokkos::parallel_for(
          "Adia slip wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn = (q1(i, j, 1) * nx(i, j) + q1(i, j, 2) * ny(i, j) +
                            q1(i, j, 3) * nz(i, j)) *
                           dplus;
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx(i, j) * dplus;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny(i, j) * dplus;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz(i, j) * dplus;

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
    threeDsubview dqdx0 = getHaloSlice(b.dqdx, face._nface, s0);
    threeDsubview dqdy0 = getHaloSlice(b.dqdy, face._nface, s0);
    threeDsubview dqdz0 = getHaloSlice(b.dqdz, face._nface, s0);

    threeDsubview dqdx1 = getHaloSlice(b.dqdx, face._nface, s1);
    threeDsubview dqdy1 = getHaloSlice(b.dqdy, face._nface, s1);
    threeDsubview dqdz1 = getHaloSlice(b.dqdz, face._nface, s1);

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
    double dplus = plus;
    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;
      s2 += plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);

      Kokkos::parallel_for(
          "Adia moving wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn = (q1(i, j, 1) * nx(i, j) + q1(i, j, 2) * ny(i, j) +
                            q1(i, j, 3) * nz(i, j)) *
                           dplus;
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx(i, j) * dplus;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny(i, j) * dplus;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz(i, j) * dplus;

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

    threeDsubview q1 = getHaloSlice(b.q, face._nface, s1);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);
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

    threeDsubview dqdx0 = getHaloSlice(b.dqdx, face._nface, s0);
    threeDsubview dqdy0 = getHaloSlice(b.dqdy, face._nface, s0);
    threeDsubview dqdz0 = getHaloSlice(b.dqdz, face._nface, s0);

    threeDsubview dqdx1 = getHaloSlice(b.dqdx, face._nface, s1);
    threeDsubview dqdy1 = getHaloSlice(b.dqdy, face._nface, s1);
    threeDsubview dqdz1 = getHaloSlice(b.dqdz, face._nface, s1);

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
    double dplus = plus;
    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);

      Kokkos::parallel_for(
          "isoT no slip wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn = (q1(i, j, 1) * nx(i, j) + q1(i, j, 2) * ny(i, j) +
                            q1(i, j, 3) * nz(i, j)) *
                           dplus;
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx(i, j) * dplus;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny(i, j) * dplus;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz(i, j) * dplus;

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

    threeDsubview q1 = getHaloSlice(b.q, face._nface, s1);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);

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

    threeDsubview dqdx0 = getHaloSlice(b.dqdx, face._nface, s0);
    threeDsubview dqdy0 = getHaloSlice(b.dqdy, face._nface, s0);
    threeDsubview dqdz0 = getHaloSlice(b.dqdz, face._nface, s0);

    threeDsubview dqdx1 = getHaloSlice(b.dqdx, face._nface, s1);
    threeDsubview dqdy1 = getHaloSlice(b.dqdy, face._nface, s1);
    threeDsubview dqdz1 = getHaloSlice(b.dqdz, face._nface, s1);

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
    double dplus = plus;
    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);

      Kokkos::parallel_for(
          "isoT slip wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // flip velo on wall
            double uDotn = (q1(i, j, 1) * nx(i, j) + q1(i, j, 2) * ny(i, j) +
                            q1(i, j, 3) * nz(i, j)) *
                           dplus;
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx(i, j) * dplus;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny(i, j) * dplus;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz(i, j) * dplus;

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

    threeDsubview dqdx0 = getHaloSlice(b.dqdx, face._nface, s0);
    threeDsubview dqdy0 = getHaloSlice(b.dqdy, face._nface, s0);
    threeDsubview dqdz0 = getHaloSlice(b.dqdz, face._nface, s0);

    threeDsubview dqdx1 = getHaloSlice(b.dqdx, face._nface, s1);
    threeDsubview dqdy1 = getHaloSlice(b.dqdy, face._nface, s1);
    threeDsubview dqdz1 = getHaloSlice(b.dqdz, face._nface, s1);

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
    double dplus = plus;
    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);

      Kokkos::parallel_for(
          "Iso T moving wall euler terms", range_face,
          KOKKOS_LAMBDA(const int i, const int j) {
            // match pressure
            q0(i, j, 0) = q1(i, j, 0);

            // mirror velo on wall
            double uDotn = (q1(i, j, 1) * nx(i, j) + q1(i, j, 2) * ny(i, j) +
                            q1(i, j, 3) * nz(i, j)) *
                           dplus;
            q0(i, j, 1) = q1(i, j, 1) - 2.0 * uDotn * nx(i, j) * dplus;
            q0(i, j, 2) = q1(i, j, 2) - 2.0 * uDotn * ny(i, j) * dplus;
            q0(i, j, 3) = q1(i, j, 3) - 2.0 * uDotn * nz(i, j) * dplus;

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

    threeDsubview q1 = getHaloSlice(b.q, face._nface, s1);
    MDRange2 range_face = MDRange2({0, 0}, {q1.extent(0), q1.extent(1)});
    for (int g = 0; g < b.ng; g++) {
      s0 -= plus * g;

      threeDsubview q0 = getHaloSlice(b.q, face._nface, s0);

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

    threeDsubview dqdx0 = getHaloSlice(b.dqdx, face._nface, s0);
    threeDsubview dqdy0 = getHaloSlice(b.dqdy, face._nface, s0);
    threeDsubview dqdz0 = getHaloSlice(b.dqdz, face._nface, s0);

    threeDsubview dqdx1 = getHaloSlice(b.dqdx, face._nface, s1);
    threeDsubview dqdy1 = getHaloSlice(b.dqdy, face._nface, s1);
    threeDsubview dqdz1 = getHaloSlice(b.dqdz, face._nface, s1);

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
