#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include <vector>
#include <math.h>

void apply_flux(std::vector<block_> mb) {
for(block_ b : mb){

  // integration cc range
  MDRange4 range({1,1,1,0},{b.ni,b.nj,b.nk,b.ne});
  Kokkos::parallel_for("Apply current fluxes to RHS",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k,
                                     const int l) {

    // Add fluxes to RHS
    b.dQ(i,j,k,l) += b.iF(i  ,j,k,l) + b.jF(i,j  ,k,l) + b.kF(i,j,k  ,l);
    b.dQ(i,j,k,l) -= b.iF(i+1,j,k,l) + b.jF(i,j+1,k,l) + b.kF(i,j,k+1,l);

  });

}};
