#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"

void dqdxyz(block_ b) {

//-------------------------------------------------------------------------------------------|
// Spatial derivatices of primative variables
//-------------------------------------------------------------------------------------------|
  MDRange4 range({1,1,1,0},{b.ni,b.nj,b.nk,b.ne});
  Kokkos::parallel_for("Apply current fluxes to RHS",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k,
                                     const int l) {

    b.dqdx(i,j,k,l) = 0.5*( b.q(i+1,j  ,k  ,l) - b.q(i-1,j  ,k  ,l) ) * b.dEdx(i,j,k) +
                      0.5*( b.q(i  ,j+1,k  ,l) - b.q(i  ,j-1,k  ,l) ) * b.dNdx(i,j,k) +
                      0.5*( b.q(i  ,j  ,k+1,l) - b.q(i  ,j  ,k-1,l) ) * b.dXdx(i,j,k) ;

    b.dqdy(i,j,k,l) = 0.5*( b.q(i+1,j  ,k  ,l) - b.q(i-1,j  ,k  ,l) ) * b.dEdy(i,j,k) +
                      0.5*( b.q(i  ,j+1,k  ,l) - b.q(i  ,j-1,k  ,l) ) * b.dNdy(i,j,k) +
                      0.5*( b.q(i  ,j  ,k+1,l) - b.q(i  ,j  ,k-1,l) ) * b.dXdy(i,j,k) ;

    b.dqdz(i,j,k,l) = 0.5*( b.q(i+1,j  ,k  ,l) - b.q(i-1,j  ,k  ,l) ) * b.dEdz(i,j,k) +
                      0.5*( b.q(i  ,j+1,k  ,l) - b.q(i  ,j-1,k  ,l) ) * b.dNdz(i,j,k) +
                      0.5*( b.q(i  ,j  ,k+1,l) - b.q(i  ,j  ,k-1,l) ) * b.dXdz(i,j,k) ;
  });

};
