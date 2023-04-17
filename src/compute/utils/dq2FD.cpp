#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include "kokkosTypes.hpp"

void dq2FD(block_ &b) {

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
