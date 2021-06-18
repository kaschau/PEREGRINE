#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include <iostream>
#include <math.h>

void advective(block_ b) {

  // i flux face range
  MDRange4 range_c({1,1,1,0},{b.ni+1,b.nj,b.nk,b.ne});
  Kokkos::parallel_for("Grid Metrics", range_c, KOKKOS_LAMBDA(const int i,
                                                              const int j,
                                                              const int k,
                                                              const int l) {
    double U;
    double uf;
    double vf;
    double wf;

    double ut;

    // Compute face normal volume flux vector
    uf = 0.5*(b.q(i,k,l,1)+b.q(i-1,j,k,1));
    vf = 0.5*(b.q(i,k,l,2)+b.q(i-1,j,k,2));
    wf = 0.5*(b.q(i,k,l,3)+b.q(i-1,j,k,3));

    U = sqrt(pow(b.isx(i,j,k)*uf,2.0) +
             pow(b.isy(i,j,k)*vf,2.0) +
             pow(b.isz(i,j,k)*wf,2.0) );

    //Compute fluxes

    // Continuity
    b.iF(i,j,k,0) = 0.5*(b.Q(i,j,k,0)+b.Q(i,j,k,0))*U;

    // x momentum rho*u*U+p
    b.iF(i,j,k,1) = 0.5*(b.Q(i,j,k,0)+b.Q(i-1,j,k,0)) * 0.5*(b.q(i,j,k,1)+b.q(i-1,j,k,1)) * U
                  + 0.5*(b.q(i,j,k,0)-b.q(i-1,j,k,0)) *      b.isx(i,j,k)                     ;

    // y momentum rho*v*U
    b.iF(i,j,k,2) = 0.5*(b.Q(i,j,k,0)+b.Q(i-1,j,k,0)) * 0.5*(b.q(i,j,k,2)+b.q(i-1,j,k,2)) * U ;

    // w momentum rho*w*U
    b.iF(i,j,k,3) = 0.5*(b.Q(i,j,k,0)+b.Q(i-1,j,k,0)) * 0.5*(b.q(i,j,k,3)+b.q(i-1,j,k,3)) * U ;

    // Total energy (rhoE+P)*U)
    b.iF(i,j,k,4) =(0.5*(b.Q(i,j,k,4)+b.Q(i-1,j,k,4)) + 0.5*(b.q(i,j,k,0)+b.q(i-1,j,k,0)))* U ;

  });

}
