#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thermdat_.hpp"
#include <vector>

void diffusive(std::vector<block_> mb, thermdat_ th) {
for(block_ b : mb){

//-------------------------------------------------------------------------------------------|
// i flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_i({1,1,1},{b.ni+1,b.nj,b.nk});
  Kokkos::parallel_for("i face visc fluxes", range_i, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {


//    // Species
//    for (int n=0; n<th.ns-1; n++)
//    {
//      b.iF(i,j,k,5+n) = rho * 0.5*(b.q(i,j,k,5+n)+b.q(i-1,j,k,5+n)) * U;
//      //b.iF(i,j,k,5+n) = 0.5*(b.Q(i,j,k,5+n)+b.Q(i-1,j,k,5+n)) * U;
//    }

  });

//-------------------------------------------------------------------------------------------|
// j flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_j({1,1,1},{b.ni,b.nj+1,b.nk});
  Kokkos::parallel_for("j face visc fluxes", range_j, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {




//    // Species
//    for (int n=0; n<th.ns-1; n++)
//    {
//      b.jF(i,j,k,5+n) = rho * 0.5*(b.q(i,j,k,5+n)+b.q(i,j-1,k,5+n)) * V;
//      //b.jF(i,j,k,5+n) = 0.5*(b.Q(i,j,k,5+n)+b.Q(i,j-1,k,5+n)) * V;
//    }

  });

//-------------------------------------------------------------------------------------------|
// k flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_k({1,1,1},{b.ni,b.nj,b.nk+1});
  Kokkos::parallel_for("k face visc fluxes", range_k, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {




//    // Species
//    for (int n=0; n<th.ns-1; n++)
//    {
//      b.kF(i,j,k,5+n) = rho * 0.5*(b.q(i,j,k,5+n)+b.q(i,j,k-1,5+n)) * W;
//      //b.kF(i,j,k,5+n) = 0.5*(b.Q(i,j,k,5+n)+b.Q(i,j,k-1,5+n)) * W;
//    }

  });


//-------------------------------------------------------------------------------------------|
// Apply fluxes to cc range
//-------------------------------------------------------------------------------------------|
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

    // Divide by cell volume
    b.dQ(i,j,k,l) /= b.J(i,j,k);

  });

}};
