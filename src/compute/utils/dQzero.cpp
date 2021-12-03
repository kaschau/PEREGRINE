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

  policy p(nblks, Kokkos::AUTO());
  Kokkos::parallel_for(
      "test", p, KOKKOS_LAMBDA(policy::member_type member) {
        int nblki = member.league_rank();

        const int ni = mb[nblki].ni;
        const int nj = mb[nblki].nj;
        const int nk = mb[nblki].nk;
        const int nl = mb[nblki].ne;

        int nijkl = (ni - 1) * (nj - 1) * (nk - 1) * nl;

        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, 0, nijkl), [=](const int &ijkl) {

              const int i =
                  mb[nblki].ng + ijkl / ((nj - 1) * (nk - 1) * nl);
              const int j =
                  mb[nblki].ng + (ijkl % ((nj - 1) * (nk - 1) * nl)) /
                                     ((nk - 1) * nl);
              const int k = mb[nblki].ng + (ijkl % ((nk - 1) * nl)) / nl;
              const int l = ijkl % nl;

              mb[nblki].dQ(i, j, k, l) = 0.0;
            });
      });
}
