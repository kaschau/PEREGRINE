#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkos_types.hpp"
#include <vector>

void dQzero(std::vector<block_> mb) {

  //-------------------------------------------------------------------------------------------|
  // Zero out dQ
  //-------------------------------------------------------------------------------------------|
  int nblks = mb.size();

  policy p(nblks, Kokkos::AUTO());
  Kokkos::parallel_for(
      "test", p, KOKKOS_LAMBDA(policy::member_type member) {
        int nblki = member.league_rank();
        block_ b = mb[nblki];

        MDRange4 range_cc(
            {b.ng, b.ng, b.ng, 0},
            {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1, b.ne});

        Kokkos::parallel_for(
            "Apply current fluxes to RHS", range_cc,
            KOKKOS_LAMBDA(const int i, const int j, const int k, const int l) {
              b.dQ(i, j, k, l) = 0.0;
            });
      });
}
