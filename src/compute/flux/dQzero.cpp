#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include <vector>

void dQzero(std::vector<block_> mb) {
for(block_ b : mb){

//-------------------------------------------------------------------------------------------|
// Zero out dQ
//-------------------------------------------------------------------------------------------|
  MDRange4 range({1,1,1,0},{b.ni,b.nj,b.nk,b.ne});
  Kokkos::parallel_for("Apply current fluxes to RHS",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k,
                                     const int l) {

    b.dQ(i,j,k,l) = 0.0;

  });

}};
