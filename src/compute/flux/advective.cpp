#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thermdat_.hpp"
#include <vector>

void advective(std::vector<block_> mb, thermdat_ th) {
for(block_ b : mb){

//-------------------------------------------------------------------------------------------|
// i flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_i({1,1,1},{b.ni+1,b.nj,b.nk});
  Kokkos::parallel_for("i face conv fluxes", range_i, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {
    double U;
    double uf;
    double vf;
    double wf;

    // Compute face normal volume flux vector
    uf = 0.5*(b.q(i,j,k,1)+b.q(i-1,j,k,1));
    vf = 0.5*(b.q(i,j,k,2)+b.q(i-1,j,k,2));
    wf = 0.5*(b.q(i,j,k,3)+b.q(i-1,j,k,3));

    U = b.isx(i,j,k)*uf +
        b.isy(i,j,k)*vf +
        b.isz(i,j,k)*wf ;

    //Compute fluxes
    double rho;
    rho = 0.5*(b.Q(i,j,k,0)+b.Q(i-1,j,k,0));

    // Continuity rho*Ui
    b.iF(i,j,k,0) = rho * U;

    // x momentum rho*u*Ui+ p*Ax
    b.iF(i,j,k,1) =                               rho * 0.5*(b.q(i,j,k,1)+b.q(i-1,j,k,1)) * U
                  + 0.5*(b.q(i,j,k,0)+b.q(i-1,j,k,0)) *      b.isx(i,j,k)                   ;

    // y momentum rho*v*Ui+ p*Ay
    b.iF(i,j,k,2) =                               rho * 0.5*(b.q(i,j,k,2)+b.q(i-1,j,k,2)) * U
                  + 0.5*(b.q(i,j,k,0)+b.q(i-1,j,k,0)) *      b.isy(i,j,k)                   ;

    // w momentum rho*w*Ui+ p*Az
    b.iF(i,j,k,3) =                               rho * 0.5*(b.q(i,j,k,3)+b.q(i-1,j,k,3)) * U
                  + 0.5*(b.q(i,j,k,0)+b.q(i-1,j,k,0)) *      b.isz(i,j,k)                   ;

    // Total energy (rhoE+ p)*Ui)
    double e;
    double em;

    e = b.Q(i  ,j,k,4)/b.Q(i  ,j,k,0) - 0.5*(pow(b.q(i  ,j,k,1),2.0) +
                                             pow(b.q(i  ,j,k,2),2.0) +
                                             pow(b.q(i  ,j,k,3),2.0));

    em= b.Q(i-1,j,k,4)/b.Q(i-1,j,k,0) - 0.5*(pow(b.q(i-1,j,k,1),2.0) +
                                             pow(b.q(i-1,j,k,2),2.0) +
                                             pow(b.q(i-1,j,k,3),2.0));

    b.iF(i,j,k,4) =( rho *(0.5*(  e         +  em         )
                         + 0.5*(b.q(i,j,k,1)*b.q(i-1,j,k,1)  +
                                b.q(i,j,k,2)*b.q(i-1,j,k,2)  +
                                b.q(i,j,k,3)*b.q(i-1,j,k,3)) ) ) * U;

    b.iF(i,j,k,4)+= 0.5*(b.q(i-1,j,k,0)*(b.q(i  ,j,k,1)*b.isx(i,j,k)
                                        +b.q(i  ,j,k,2)*b.isy(i,j,k)
                                        +b.q(i  ,j,k,3)*b.isz(i,j,k) ) +
                         b.q(i  ,j,k,0)*(b.q(i-1,j,k,1)*b.isx(i,j,k)
                                        +b.q(i-1,j,k,2)*b.isy(i,j,k)
                                        +b.q(i-1,j,k,3)*b.isz(i,j,k) ) );
    // Species
    for (int n=0; n<th.ns-1; n++)
    {
      b.iF(i,j,k,5+n) = rho * 0.5*(b.q(i,j,k,5+n)+b.q(i-1,j,k,5+n)) * U;
      //b.iF(i,j,k,5+n) = 0.5*(b.Q(i,j,k,5+n)+b.Q(i-1,j,k,5+n)) * U;
    }

  });

//-------------------------------------------------------------------------------------------|
// j flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_j({1,1,1},{b.ni,b.nj+1,b.nk});
  Kokkos::parallel_for("j face conv fluxes", range_j, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {

    double V;
    double uf;
    double vf;
    double wf;

    // Compute face normal volume flux vector
    uf = 0.5*(b.q(i,j,k,1)+b.q(i,j-1,k,1));
    vf = 0.5*(b.q(i,j,k,2)+b.q(i,j-1,k,2));
    wf = 0.5*(b.q(i,j,k,3)+b.q(i,j-1,k,3));

    V = b.jsx(i,j,k)*uf +
        b.jsy(i,j,k)*vf +
        b.jsz(i,j,k)*wf ;

    //Compute fluxes
    double rho;
    rho = 0.5*(b.Q(i,j,k,0)+b.Q(i,j-1,k,0));

    // Continuity rho*Vj
    b.jF(i,j,k,0) = rho * V;

    // x momentum rho*u*Vj+ pAx
    b.jF(i,j,k,1) =                               rho * 0.5*(b.q(i,j,k,1)+b.q(i,j-1,k,1)) * V
                  + 0.5*(b.q(i,j,k,0)+b.q(i,j-1,k,0)) *      b.jsx(i,j,k)                   ;

    // y momentum rho*v*Vj+ pAy
    b.jF(i,j,k,2) =                               rho * 0.5*(b.q(i,j,k,2)+b.q(i,j-1,k,2)) * V
                  + 0.5*(b.q(i,j,k,0)+b.q(i,j-1,k,0)) *      b.jsy(i,j,k)                   ;

    // w momentum rho*w*Vj+ pAz
    b.jF(i,j,k,3) =                               rho * 0.5*(b.q(i,j,k,3)+b.q(i,j-1,k,3)) * V
                  + 0.5*(b.q(i,j,k,0)+b.q(i,j-1,k,0)) *      b.jsz(i,j,k)                   ;

    // Total energy (rhoE+P)*Vj)
    double e;
    double em;

    e = b.Q(i,j  ,k,4)/b.Q(i,j  ,k,0) - 0.5*(pow(b.q(i,j  ,k,1),2.0) +
                                             pow(b.q(i,j  ,k,2),2.0) +
                                             pow(b.q(i,j  ,k,3),2.0));

    em= b.Q(i,j-1,k,4)/b.Q(i,j-1,k,0) - 0.5*(pow(b.q(i,j-1,k,1),2.0) +
                                             pow(b.q(i,j-1,k,2),2.0) +
                                             pow(b.q(i,j-1,k,3),2.0));

    b.jF(i,j,k,4) =( rho *(0.5*(  e         +  em         )
                         + 0.5*(b.q(i,j,k,1)*b.q(i,j-1,k,1)  +
                                b.q(i,j,k,2)*b.q(i,j-1,k,2)  +
                                b.q(i,j,k,3)*b.q(i,j-1,k,3)) ) ) * V;

    b.jF(i,j,k,4)+= 0.5*(b.q(i,j-1,k,0)*(b.q(i,j  ,k,1)*b.jsx(i,j,k)
                                        +b.q(i,j  ,k,2)*b.jsy(i,j,k)
                                        +b.q(i,j  ,k,3)*b.jsz(i,j,k) ) +
                         b.q(i,j  ,k,0)*(b.q(i,j-1,k,1)*b.jsx(i,j,k)
                                        +b.q(i,j-1,k,2)*b.jsy(i,j,k)
                                        +b.q(i,j-1,k,3)*b.jsz(i,j,k) ) );
    // Species
    for (int n=0; n<th.ns-1; n++)
    {
      b.jF(i,j,k,5+n) = rho * 0.5*(b.q(i,j,k,5+n)+b.q(i,j-1,k,5+n)) * V;
      //b.jF(i,j,k,5+n) = 0.5*(b.Q(i,j,k,5+n)+b.Q(i,j-1,k,5+n)) * V;
    }

  });

//-------------------------------------------------------------------------------------------|
// k flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_k({1,1,1},{b.ni,b.nj,b.nk+1});
  Kokkos::parallel_for("k face conv fluxes", range_k, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {

    double W;
    double uf;
    double vf;
    double wf;

    // Compute face normal volume flux vector
    uf = 0.5*(b.q(i,j,k,1)+b.q(i,j,k-1,1));
    vf = 0.5*(b.q(i,j,k,2)+b.q(i,j,k-1,2));
    wf = 0.5*(b.q(i,j,k,3)+b.q(i,j,k-1,3));

    W = b.ksx(i,j,k)*uf +
        b.ksy(i,j,k)*vf +
        b.ksz(i,j,k)*wf ;

    //Compute fluxes
    double rho;
    rho = 0.5*(b.Q(i,j,k,0)+b.Q(i,j,k-1,0));
    // Continuity rho*Wk
    b.kF(i,j,k,0) = rho * W;

    // x momentum rho*u*Wk+ pAx
    b.kF(i,j,k,1) =                               rho * 0.5*(b.q(i,j,k,1)+b.q(i,j,k-1,1)) * W
                  + 0.5*(b.q(i,j,k,0)+b.q(i,j,k-1,0)) *      b.ksx(i,j,k)                   ;

    // y momentum rho*v*Wk+ pAy
    b.kF(i,j,k,2) =                               rho * 0.5*(b.q(i,j,k,2)+b.q(i,j,k-1,2)) * W
                  + 0.5*(b.q(i,j,k,0)+b.q(i,j,k-1,0)) *      b.ksy(i,j,k)                   ;

    // w momentum rho*w*Wk+ pAz
    b.kF(i,j,k,3) =                               rho * 0.5*(b.q(i,j,k,3)+b.q(i,j,k-1,3)) * W
                  + 0.5*(b.q(i,j,k,0)+b.q(i,j,k-1,0)) *      b.ksz(i,j,k)                   ;

    // Total energy (rhoE+P)*Wk)
    double e;
    double em;

    e = b.Q(i,j,k  ,4)/b.Q(i,j,k  ,0) - 0.5*(pow(b.q(i,j,k  ,1),2.0) +
                                             pow(b.q(i,j,k  ,2),2.0) +
                                             pow(b.q(i,j,k  ,3),2.0));

    em= b.Q(i,j,k-1,4)/b.Q(i,j,k-1,0) - 0.5*(pow(b.q(i,j,k-1,1),2.0) +
                                             pow(b.q(i,j,k-1,2),2.0) +
                                             pow(b.q(i,j,k-1,3),2.0));

    b.kF(i,j,k,4) =( rho *(0.5*(  e         +  em         )
                         + 0.5*(b.q(i,j,k,1)*b.q(i,j,k-1,1)  +
                                b.q(i,j,k,2)*b.q(i,j,k-1,2)  +
                                b.q(i,j,k,3)*b.q(i,j,k-1,3)) ) ) * W;

    b.kF(i,j,k,4)+= 0.5*(b.q(i,j,k-1,0)*(b.q(i,j,k  ,1)*b.ksx(i,j,k)
                                        +b.q(i,j,k  ,2)*b.ksy(i,j,k)
                                        +b.q(i,j,k  ,3)*b.ksz(i,j,k) ) +
                         b.q(i,j,k  ,0)*(b.q(i,j,k-1,1)*b.ksx(i,j,k)
                                        +b.q(i,j,k-1,2)*b.ksy(i,j,k)
                                        +b.q(i,j,k-1,3)*b.ksz(i,j,k) ) );
    // Species
    for (int n=0; n<th.ns-1; n++)
    {
      b.kF(i,j,k,5+n) = rho * 0.5*(b.q(i,j,k,5+n)+b.q(i,j,k-1,5+n)) * W;
      //b.kF(i,j,k,5+n) = 0.5*(b.Q(i,j,k,5+n)+b.Q(i,j,k-1,5+n)) * W;
    }

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
    b.dQ(i,j,k,l) += ( b.iF(i  ,j,k,l) + b.jF(i,j  ,k,l) + b.kF(i,j,k  ,l) ) / b.J(i,j,k);
    b.dQ(i,j,k,l) -= ( b.iF(i+1,j,k,l) + b.jF(i,j+1,k,l) + b.kF(i,j,k+1,l) ) / b.J(i,j,k);

  });

}};
