#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkos_types.hpp"
#include <Kokkos_Macros.hpp>
#include <vector>

void dQzero(std::vector<block_> mb) {

  //-------------------------------------------------------------------------------------------|
  // Zero out dQ
  //-------------------------------------------------------------------------------------------|
  int nblks = mb.size();

  Kokkos::View<block_ *, exec_space> mbv("test", nblks);
  for (int b = 0; b < nblks; b++) {
    mbv(b) = mb[b];
  }

  policy p(nblks, Kokkos::AUTO());
  Kokkos::parallel_for(
      "test", p, KOKKOS_LAMBDA(policy::member_type member) {
        int nblki = member.league_rank();

        const int ni = mbv(nblki).ni;
        const int nj = mbv(nblki).nj;
        const int nk = mbv(nblki).nk;
        const int nl = mbv(nblki).ne;
        const int ng = mbv(nblki).ng;

        int nijkl = (ni - 1) * (nj - 1) * (nk - 1) * nl;

        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, 0, nijkl), [=](const int &ijkl) {
              const int i = ng + ijkl / ((nj - 1) * (nk - 1) * nl);
              const int j =
                  ng + (ijkl % ((nj - 1) * (nk - 1) * nl)) / ((nk - 1) * nl);
              const int k = ng + (ijkl % ((nk - 1) * nl)) / nl;
              const int l = ijkl % nl;

              mbv(nblki).dQ(i, j, k, l) = 0.0;
            });
      });
}
