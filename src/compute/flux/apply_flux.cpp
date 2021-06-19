#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include <iostream>
#include <math.h>

void apply_flux(block_ b) {

  // i flux face range
  MDRange3 range({1,1,1},{b.ni,b.nj,b.nk});
  Kokkos::parallel_for("Apply current fluxes to LHS",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

    // Continuity
    b.dQ(i,j,k,0) += b.iF(i  ,j,k,0) + b.jF(i,j  ,k,0) + b.kF(i,j,k  ,0);
    b.dQ(i,j,k,0) -= b.iF(i+1,j,k,0) + b.jF(i,j+1,k,0) + b.kF(i,j,k+1,0);

    // x momentum
    b.dQ(i,j,k,1) += b.iF(i  ,j,k,1) + b.jF(i,j  ,k,1) + b.kF(i,j,k  ,1);
    b.dQ(i,j,k,1) -= b.iF(i+1,j,k,1) + b.jF(i,j+1,k,1) + b.kF(i,j,k+1,1);

    // y momentum
    b.dQ(i,j,k,2) += b.iF(i  ,j,k,2) + b.jF(i,j  ,k,2) + b.kF(i,j,k  ,2);
    b.dQ(i,j,k,2) -= b.iF(i+1,j,k,2) + b.jF(i,j+1,k,2) + b.kF(i,j,k+1,2);

    // w momentum
    b.dQ(i,j,k,3) += b.iF(i  ,j,k,3) + b.jF(i,j  ,k,3) + b.kF(i,j,k  ,3);
    b.dQ(i,j,k,3) -= b.iF(i+1,j,k,3) + b.jF(i,j+1,k,3) + b.kF(i,j,k+1,3);

    // Total energy
    b.dQ(i,j,k,4) += b.iF(i  ,j,k,4) + b.jF(i,j  ,k,4) + b.kF(i,j,k  ,4);
    b.dQ(i,j,k,4) -= b.iF(i+1,j,k,4) + b.jF(i,j+1,k,4) + b.kF(i,j,k+1,4);

  });

}
