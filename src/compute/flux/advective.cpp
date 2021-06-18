#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include <iostream>
#include <math.h>

void advective(block_ b) {

  // i flux face range
  MDRange3 range_i({1,1,1},{b.ni+1,b.nj,b.nk});
  Kokkos::parallel_for("Grid Metrics", range_i, KOKKOS_LAMBDA(const int i,
                                                              const int j,
                                                              const int k) {
    double U;
    double uf;
    double vf;
    double wf;

    double ut;

    // Compute face normal volume flux vector
    uf = 0.5*(b.q(i,j,k,1)+b.q(i-1,j,k,1));
    vf = 0.5*(b.q(i,j,k,2)+b.q(i-1,j,k,2));
    wf = 0.5*(b.q(i,j,k,3)+b.q(i-1,j,k,3));

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

  // j flux face range
  MDRange3 range_j({1,1,1},{b.ni,b.nj+1,b.nk});
  Kokkos::parallel_for("Grid Metrics", range_j, KOKKOS_LAMBDA(const int i,
                                                              const int j,
                                                              const int k) {

    double V;
    double uf;
    double vf;
    double wf;

    double ut;

    // Compute face normal volume flux vector
    uf = 0.5*(b.q(i,j,k,1)+b.q(i,j-1,k,1));
    vf = 0.5*(b.q(i,j,k,2)+b.q(i,j-1,k,2));
    wf = 0.5*(b.q(i,j,k,3)+b.q(i,j-1,k,3));

    V = sqrt(pow(b.jsx(i,j,k)*uf,2.0) +
             pow(b.jsy(i,j,k)*vf,2.0) +
             pow(b.jsz(i,j,k)*wf,2.0) );

    //Compute fluxes

    // Continuity
    b.jF(i,j,k,0) = 0.5*(b.Q(i,j,k,0)+b.Q(i,j,k,0))*V;

    // x momentum rho*u*V
    b.jF(i,j,k,1) = 0.5*(b.Q(i,j,k,0)+b.Q(i,j-1,k,0)) * 0.5*(b.q(i,j,k,1)+b.q(i,j-1,k,1)) * V ;

    // y momentum rho*v*V+p
    b.jF(i,j,k,2) = 0.5*(b.Q(i,j,k,0)+b.Q(i,j-1,k,0)) * 0.5*(b.q(i,j,k,2)+b.q(i,j-1,k,2)) * V
                  + 0.5*(b.q(i,j,k,0)-b.q(i,j-1,k,0)) *      b.jsx(i,j,k)                     ;

    // w momentum rho*w*V
    b.jF(i,j,k,3) = 0.5*(b.Q(i,j,k,0)+b.Q(i,j-1,k,0)) * 0.5*(b.q(i,j,k,3)+b.q(i,j-1,k,3)) * V ;

    // Total energy (rhoE+P)*V)
    b.jF(i,j,k,4) =(0.5*(b.Q(i,j,k,4)+b.Q(i,j-1,k,4)) + 0.5*(b.q(i,j,k,0)+b.q(i,j-1,k,0)))* V ;

  });

  // k flux face range
  MDRange3 range_k({1,1,1},{b.ni,b.nj,b.nk+1});
  Kokkos::parallel_for("Grid Metrics", range_k, KOKKOS_LAMBDA(const int i,
                                                              const int j,
                                                              const int k) {

    double W;
    double uf;
    double vf;
    double wf;

    double ut;

    // Compute face normal volume flux vector
    uf = 0.5*(b.q(i,j,k,1)+b.q(i,j,k-1,1));
    vf = 0.5*(b.q(i,j,k,2)+b.q(i,j,k-1,2));
    wf = 0.5*(b.q(i,j,k,3)+b.q(i,j,k-1,3));

    W = sqrt(pow(b.ksx(i,j,k)*uf,2.0) +
             pow(b.ksy(i,j,k)*vf,2.0) +
             pow(b.ksz(i,j,k)*wf,2.0) );

    //Compute fluxes

    // Continuity
    b.kF(i,j,k,0) = 0.5*(b.Q(i,j,k,0)+b.Q(i,j,k,0))*W;

    // x momentum rho*u*W
    b.kF(i,j,k,1) = 0.5*(b.Q(i,j,k,0)+b.Q(i,j,k-1,0)) * 0.5*(b.q(i,j,k,1)+b.q(i,j,k-1,1)) * W ;

    // y momentum rho*v*W
    b.kF(i,j,k,2) = 0.5*(b.Q(i,j,k,0)+b.Q(i,j,k-1,0)) * 0.5*(b.q(i,j,k,2)+b.q(i,j,k-1,2)) * W ;

    // w momentum rho*w*W+p
    b.kF(i,j,k,3) = 0.5*(b.Q(i,j,k,0)+b.Q(i,j,k-1,0)) * 0.5*(b.q(i,j,k,3)+b.q(i,j,k-1,3)) * W ;
                  + 0.5*(b.q(i,j,k,0)-b.q(i,j,k-1,0)) *      b.ksx(i,j,k)                     ;

    // Total energy (rhoE+P)*W)
    b.kF(i,j,k,4) =(0.5*(b.Q(i,j,k,4)+b.Q(i,j,k-1,4)) + 0.5*(b.q(i,j,k,0)+b.q(i,j,k-1,0)))* W ;

  });

}
