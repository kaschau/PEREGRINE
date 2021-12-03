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

  Kokkos::View<block_*, exec_space> mbD("testD", nblks);
  Kokkos::View<block_*, Kokkos::HostSpace> mbH("testH", nblks);
  for (int b = 0; b < nblks; b++) {
    mbH(b) = mb[b];
  }
  Kokkos::deep_copy(mbD, mbH);

  policy p(nblks, Kokkos::AUTO());
  Kokkos::parallel_for(
      "test", p, KOKKOS_LAMBDA(policy::member_type member) {

        int nblki = member.league_rank();
        auto& b = mbD(nblki);

        int nijkl = (b.ni - 1) * (b.nj - 1) * (b.nk - 1) * b.ne;

        Kokkos::parallel_for(
            Kokkos::TeamVectorRange(member, 0, nijkl), [=](const int &ijkl) {
              const int i = b.ng + ijkl / ((b.nj - 1) * (b.nk - 1) * b.ne);
              const int j =
                  b.ng + (ijkl % ((b.nj - 1) * (b.nk - 1) * b.ne)) / ((b.nk - 1) * b.ne);
              const int k = b.ng + (ijkl % ((b.nk - 1) * b.ne)) / b.ne;
              const int l = ijkl % b.ne;

              b.dQ(i, j, k, l) = 0.0;
            });
      });
}
