#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include "kokkos_types.hpp"

void dq2FD(block_ b) {

  //-------------------------------------------------------------------------------------------|
  // Spatial derivatices of primative variables
  // estimated via second order finite difference
  //-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng, 0},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
  Kokkos::parallel_for(
      "2nd order spatial deriv", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
        double dqdE = 0.5 * (b.q(i + 1, j, k, l) - b.q(i - 1, j, k, l));
        double dqdN = 0.5 * (b.q(i, j + 1, k, l) - b.q(i, j - 1, k, l));
        double dqdC = 0.5 * (b.q(i, j, k + 1, l) - b.q(i, j, k - 1, l));

        b.dqdx(i, j, k, l) = dqdE * b.dEdx(i, j, k) + dqdN * b.dNdx(i, j, k) +
                             dqdC * b.dCdx(i, j, k);

        b.dqdy(i, j, k, l) = dqdE * b.dEdy(i, j, k) + dqdN * b.dNdy(i, j, k) +
                             dqdC * b.dCdy(i, j, k);

        b.dqdz(i, j, k, l) = dqdE * b.dEdz(i, j, k) + dqdN * b.dNdz(i, j, k) +
                             dqdC * b.dCdz(i, j, k);
      });
}

void dq2FDoneSided(block_ b, int nface) {

  const int ng = b.ng;
  int s0, s1, s2, plus;
  setHaloSlices(s0, s1, s2, plus, b.ni, b.nj, b.nk, ng, nface);
  double dplus = plus;

  switch (nface) {
  case 1:
  case 2: {
    MDRange3 range_cc({b.ng, b.ng, 0},
                      {b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});
    Kokkos::parallel_for(
        "one sided deriv i", range_cc,
        KOKKOS_LAMBDA(const int j, const int k, const int l) {
          double dqdE =
              dplus * 0.5 *
              (-3.0 * b.q(s1, j, k, l) + 4.0 * b.q(s1 + plus, j, k, l) -
               b.q(s1 + plus * 2, j, k, l));

          double dqdN = 0.5 * (b.q(s1, j + 1, k, l) - b.q(s1, j - 1, k, l));
          double dqdC = 0.5 * (b.q(s1, j, k + 1, l) - b.q(s1, j, k - 1, l));

          b.dqdx(s1, j, k, l) = dqdE * b.dEdx(s1, j, k) +
                                dqdN * b.dNdx(s1, j, k) +
                                dqdC * b.dCdx(s1, j, k);

          b.dqdy(s1, j, k, l) = dqdE * b.dEdy(s1, j, k) +
                                dqdN * b.dNdy(s1, j, k) +
                                dqdC * b.dCdy(s1, j, k);

          b.dqdz(s1, j, k, l) = dqdE * b.dEdz(s1, j, k) +
                                dqdN * b.dNdz(s1, j, k) +
                                dqdC * b.dCdz(s1, j, k);
        });

    break;
  }
  case 3:
  case 4: {
    MDRange3 range_cc({b.ng, b.ng, 0},
                      {b.ni + b.ng - 1, b.nk + b.ng - 1, b.ne});

    Kokkos::parallel_for(
        "one sided deriv j", range_cc,
        KOKKOS_LAMBDA(const int i, const int k, const int l) {
          double dqdE = 0.5 * (b.q(i + 1, s1, k, l) - b.q(i - 1, s1, k, l));
          double dqdN =
              dplus * 0.5 *
              (-3.0 * b.q(i, s1, k, l) + 4.0 * b.q(i, s1 + plus, k, l) -
               b.q(i, s1 + plus * 2, k, l));
          double dqdC = 0.5 * (b.q(i, s1, k + 1, l) - b.q(i, s1, k - 1, l));

          b.dqdx(i, s1, k, l) = dqdE * b.dEdx(i, s1, k) +
                                dqdN * b.dNdx(i, s1, k) +
                                dqdC * b.dCdx(i, s1, k);

          b.dqdy(i, s1, k, l) = dqdE * b.dEdy(i, s1, k) +
                                dqdN * b.dNdy(i, s1, k) +
                                dqdC * b.dCdy(i, s1, k);

          b.dqdz(i, s1, k, l) = dqdE * b.dEdz(i, s1, k) +
                                dqdN * b.dNdz(i, s1, k) +
                                dqdC * b.dCdz(i, s1, k);
        });

    break;
  }
  case 5:
  case 6: {
    MDRange3 range_cc({b.ng, b.ng, 0},
                      {b.ni + b.ng - 1, b.nj + b.ng - 1, b.ne});

    Kokkos::parallel_for(
        "one sided deriv k", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int l) {
          double dqdE = 0.5 * (b.q(i + 1, j, s1, l) - b.q(i - 1, j, s1, l));
          double dqdN = 0.5 * (b.q(i, j + 1, s1, l) - b.q(i, j - 1, s1, l));
          double dqdC =
              dplus * 0.5 *
              (-3.0 * b.q(i, j, s1, l) + 4.0 * b.q(i, j, s1 + plus, l) -
               b.q(i, j, s1 + plus * 2, l));

          b.dqdx(i, j, s1, l) = dqdE * b.dEdx(i, j, s1) +
                                dqdN * b.dNdx(i, j, s1) +
                                dqdC * b.dCdx(i, j, s1);

          b.dqdy(i, j, s1, l) = dqdE * b.dEdy(i, j, s1) +
                                dqdN * b.dNdy(i, j, s1) +
                                dqdC * b.dCdy(i, j, s1);

          b.dqdz(i, j, s1, l) = dqdE * b.dEdz(i, j, s1) +
                                dqdN * b.dNdz(i, j, s1) +
                                dqdC * b.dCdz(i, j, s1);
        });

    break;
  }
  default:
    throw std::invalid_argument(" <-- Unknown argument to dq2FDOneSided");
  }
}
