#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"

void secondOrderFD(block_ b, const thtrdat_ th) {

//-------------------------------------------------------------------------------------------|
// i flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_i({1,1,1},{b.ni+1,b.nj,b.nk});
  Kokkos::parallel_for("i face visc fluxes", range_i, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {

    double dudx,dudy,dudz;
    double dvdx,dvdy,dvdz;
    double dwdx,dwdy,dwdz;

    double txx,txy,txz;
    double tyx,tyy,tyz;
    double tzx,tzy,tzz;

    double dTdx,dTdy,dTdz;
    double uf,vf,wf;
    double q;

    double dNdx,dNdy,dNdz;

    const double c23 = 2.0/3.0;

    double mu     = 0.5*(b.qt(i,j,k,0) + b.qt(i-1,j,k,0));
    double kappa  = 0.5*(b.qt(i,j,k,1) + b.qt(i-1,j,k,1));

    // continuity
    b.iF(i,j,k,0) = 0.0;

    // Derivatives on face
    dudx = 0.5*(b.dqdx(i,j,k,1) + b.dqdx(i-1,j,k,1) );
    dvdx = 0.5*(b.dqdx(i,j,k,2) + b.dqdx(i-1,j,k,2) );
    dwdx = 0.5*(b.dqdx(i,j,k,3) + b.dqdx(i-1,j,k,3) );

    dudy = 0.5*(b.dqdy(i,j,k,1) + b.dqdy(i-1,j,k,1) );
    dvdy = 0.5*(b.dqdy(i,j,k,2) + b.dqdy(i-1,j,k,2) );
    dwdy = 0.5*(b.dqdy(i,j,k,3) + b.dqdy(i-1,j,k,3) );

    dudz = 0.5*(b.dqdz(i,j,k,1) + b.dqdz(i-1,j,k,1) );
    dvdz = 0.5*(b.dqdz(i,j,k,2) + b.dqdz(i-1,j,k,2) );
    dwdz = 0.5*(b.dqdz(i,j,k,3) + b.dqdz(i-1,j,k,3) );

    // x momentum
    txx = c23*mu*(2.0*dudx - dvdy - dwdz);
    txy =     mu*(    dvdx + dudy       );
    txz =     mu*(    dwdx        + dudy);

    b.iF(i,j,k,1) = - ( txx * b.isx(i,j,k) +
                        txy * b.isy(i,j,k) +
                        txz * b.isz(i,j,k) );

    // y momentum
    tyx = txy;
    tyy = c23*mu*(-dudx +2.0*dvdy - dwdz);
    tyz =     mu*(           dwdy + dvdz);

    b.iF(i,j,k,2) = - ( tyx * b.isx(i,j,k) +
                        tyy * b.isy(i,j,k) +
                        tyz * b.isz(i,j,k) );

    // z momentum
    tzx = txz;
    tzy = tyz;
    tzz = c23*mu*(-dudx - dvdy +2.0*dwdz);

    b.iF(i,j,k,3) = - ( tzx * b.isx(i,j,k) +
                        tzy * b.isy(i,j,k) +
                        tzz * b.isz(i,j,k) );

    // energy
    //   heat conduction
    dTdx = 0.5 * ( b.dqdx(i,j,k,4) + b.dqdx(i-1,j,k,4) );
    dTdy = 0.5 * ( b.dqdy(i,j,k,4) + b.dqdy(i-1,j,k,4) );
    dTdz = 0.5 * ( b.dqdz(i,j,k,4) + b.dqdz(i-1,j,k,4) );

    q =  kappa*( dTdx * b.isx(i,j,k) +
                 dTdy * b.isy(i,j,k) +
                 dTdz * b.isz(i,j,k) );

    // flow work
    // Compute face normal volume flux vector
    uf = 0.5*(b.q(i,j,k,1)+b.q(i-1,j,k,1));
    vf = 0.5*(b.q(i,j,k,2)+b.q(i-1,j,k,2));
    wf = 0.5*(b.q(i,j,k,3)+b.q(i-1,j,k,3));

    b.iF(i,j,k,4) = ( uf*txx*b.isx(i,j,k) +
                      vf*txy*b.isy(i,j,k) +
                      wf*txz*b.isz(i,j,k) ) - q;

    // Species
    double Dij;
    for (int n=0; n<th.ns-1; n++)
    {
      Dij  = 0.5 * ( b.qt(i,j,k,2+n) + b.qt(i-1,j,k,2+n)     );
      dNdx = 0.5 * ( b.dqdx(i,j,k,5+n) + b.dqdx(i-1,j,k,5+n) );
      dNdy = 0.5 * ( b.dqdy(i,j,k,5+n) + b.dqdy(i-1,j,k,5+n) );
      dNdz = 0.5 * ( b.dqdz(i,j,k,5+n) + b.dqdz(i-1,j,k,5+n) );

      b.iF(i,j,k,5+n) = -Dij*( dNdx * b.isx(i,j,k) +
                               dNdy * b.isy(i,j,k) +
                               dNdz * b.isz(i,j,k) );
    }

  });

//-------------------------------------------------------------------------------------------|
// j flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_j({1,1,1},{b.ni,b.nj+1,b.nk});
  Kokkos::parallel_for("j face visc fluxes", range_j, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {

    double dudx,dudy,dudz;
    double dvdx,dvdy,dvdz;
    double dwdx,dwdy,dwdz;

    double txx,txy,txz;
    double tyx,tyy,tyz;
    double tzx,tzy,tzz;

    double dTdx,dTdy,dTdz;
    double uf,vf,wf;
    double q;

    double dNdx,dNdy,dNdz;

    const double c23 = 2.0/3.0;

    double mu     = 0.5*(b.qt(i,j,k,0) + b.qt(i,j-1,k,0));
    double kappa  = 0.5*(b.qt(i,j,k,1) + b.qt(i,j-1,k,1));

    // continuity
    b.jF(i,j,k,0) = 0.0;

    // Spatial derivative on face
    dudx = 0.5*(b.dqdx(i,j,k,1) + b.dqdx(i,j-1,k,1) );
    dvdx = 0.5*(b.dqdx(i,j,k,2) + b.dqdx(i,j-1,k,2) );
    dwdx = 0.5*(b.dqdx(i,j,k,3) + b.dqdx(i,j-1,k,3) );

    dudy = 0.5*(b.dqdy(i,j,k,1) + b.dqdy(i,j-1,k,1) );
    dvdy = 0.5*(b.dqdy(i,j,k,2) + b.dqdy(i,j-1,k,2) );
    dwdy = 0.5*(b.dqdy(i,j,k,3) + b.dqdy(i,j-1,k,3) );

    dudz = 0.5*(b.dqdz(i,j,k,1) + b.dqdz(i,j-1,k,1) );
    dvdz = 0.5*(b.dqdz(i,j,k,2) + b.dqdz(i,j-1,k,2) );
    dwdz = 0.5*(b.dqdz(i,j,k,3) + b.dqdz(i,j-1,k,3) );

    // x momentum
    txx = c23*mu*(2.0*dudx - dvdy - dwdz);
    txy =     mu*(    dvdx + dudy       );
    txz =     mu*(    dwdx        + dudy);

    b.jF(i,j,k,1) = - ( txx * b.jsx(i,j,k) +
                        txy * b.jsy(i,j,k) +
                        txz * b.jsz(i,j,k) );

    // y momentum
    tyx = txy;
    tyy = c23*mu*(-dudx +2.0*dvdy - dwdz);
    tyz =     mu*(           dwdy + dvdz);

    b.jF(i,j,k,2) = - ( tyx * b.jsx(i,j,k) +
                        tyy * b.jsy(i,j,k) +
                        tyz * b.jsz(i,j,k) );

    // z momentum
    tzx = txz;
    tzy = tyz;
    tzz = c23*mu*(-dudx - dvdy +2.0*dwdz);

    b.jF(i,j,k,3) = - ( tzx * b.jsx(i,j,k) +
                        tzy * b.jsy(i,j,k) +
                        tzz * b.jsz(i,j,k) );

    // energy
    //   heat conduction
    dTdx = 0.5 * ( b.dqdx(i,j,k,4) + b.dqdx(i,j-1,k,4) );
    dTdy = 0.5 * ( b.dqdy(i,j,k,4) + b.dqdy(i,j-1,k,4) );
    dTdz = 0.5 * ( b.dqdz(i,j,k,4) + b.dqdz(i,j-1,k,4) );

    q =  kappa*( dTdx * b.jsx(i,j,k) +
                 dTdy * b.jsy(i,j,k) +
                 dTdz * b.jsz(i,j,k) );

    // flow work
    // Compute face normal volume flux vector
    uf = 0.5*(b.q(i,j,k,1)+b.q(i,j-1,k,1));
    vf = 0.5*(b.q(i,j,k,2)+b.q(i,j-1,k,2));
    wf = 0.5*(b.q(i,j,k,3)+b.q(i,j-1,k,3));

    b.jF(i,j,k,4) = ( uf*txx*b.jsx(i,j,k) +
                      vf*txy*b.jsy(i,j,k) +
                      wf*txz*b.jsz(i,j,k) ) - q;

    // Species
    double Dij;
    for (int n=0; n<th.ns-1; n++)
    {
      Dij  = 0.5 * ( b.qt(i,j,k,2+n) + b.qt(i,j-1,k,2+n)     );
      dNdx = 0.5 * ( b.dqdx(i,j,k,5+n) + b.dqdx(i,j-1,k,5+n) );
      dNdy = 0.5 * ( b.dqdy(i,j,k,5+n) + b.dqdy(i,j-1,k,5+n) );
      dNdz = 0.5 * ( b.dqdz(i,j,k,5+n) + b.dqdz(i,j-1,k,5+n) );

      b.jF(i,j,k,5+n) = -Dij*( dNdx * b.jsx(i,j,k) +
                               dNdy * b.jsy(i,j,k) +
                               dNdz * b.jsz(i,j,k) );
    }

  });

//-------------------------------------------------------------------------------------------|
// k flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_k({1,1,1},{b.ni,b.nj,b.nk+1});
  Kokkos::parallel_for("k face visc fluxes", range_k, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {

    double dudx,dudy,dudz;
    double dvdx,dvdy,dvdz;
    double dwdx,dwdy,dwdz;

    double txx,txy,txz;
    double tyx,tyy,tyz;
    double tzx,tzy,tzz;

    double dTdx,dTdy,dTdz;
    double uf,vf,wf;
    double q;

    double dNdx,dNdy,dNdz;

    const double c23 = 2.0/3.0;

    double mu     = 0.5*(b.qt(i,j,k,0) + b.qt(i,j,k-1,0));
    double kappa  = 0.5*(b.qt(i,j,k,1) + b.qt(i,j,k-1,1));

    // continuity
    b.kF(i,j,k,0) = 0.0;

    // Spatial derivative on face
    dudx = 0.5*(b.dqdx(i,j,k,1) + b.dqdx(i,j,k-1,1) );
    dvdx = 0.5*(b.dqdx(i,j,k,2) + b.dqdx(i,j,k-1,2) );
    dwdx = 0.5*(b.dqdx(i,j,k,3) + b.dqdx(i,j,k-1,3) );

    dudy = 0.5*(b.dqdy(i,j,k,1) + b.dqdy(i,j,k-1,1) );
    dvdy = 0.5*(b.dqdy(i,j,k,2) + b.dqdy(i,j,k-1,2) );
    dwdy = 0.5*(b.dqdy(i,j,k,3) + b.dqdy(i,j,k-1,3) );

    dudz = 0.5*(b.dqdz(i,j,k,1) + b.dqdz(i,j,k-1,1) );
    dvdz = 0.5*(b.dqdz(i,j,k,2) + b.dqdz(i,j,k-1,2) );
    dwdz = 0.5*(b.dqdz(i,j,k,3) + b.dqdz(i,j,k-1,3) );

    // x momentum
    txx = c23*mu*(2.0*dudx - dvdy - dwdz);
    txy =     mu*(    dvdx + dudy       );
    txz =     mu*(    dwdx        + dudy);

    b.kF(i,j,k,1) = - ( txx * b.ksx(i,j,k) +
                        txy * b.ksy(i,j,k) +
                        txz * b.ksz(i,j,k) );

    // y momentum
    tyx = txy;
    tyy = c23*mu*(-dudx +2.0*dvdy - dwdz);
    tyz =     mu*(           dwdy + dvdz);

    b.kF(i,j,k,2) = - ( tyx * b.ksx(i,j,k) +
                        tyy * b.ksy(i,j,k) +
                        tyz * b.ksz(i,j,k) );

    // z momentum
    tzx = txz;
    tzy = tyz;
    tzz = c23*mu*(-dudx - dvdy +2.0*dwdz);

    b.kF(i,j,k,3) = - ( tzx * b.ksx(i,j,k) +
                        tzy * b.ksy(i,j,k) +
                        tzz * b.ksz(i,j,k) );

    // energy
    //   heat conduction
    dTdx = 0.5 * ( b.dqdx(i,j,k,4) + b.dqdx(i,j,k-1,4) );
    dTdy = 0.5 * ( b.dqdy(i,j,k,4) + b.dqdy(i,j,k-1,4) );
    dTdz = 0.5 * ( b.dqdz(i,j,k,4) + b.dqdz(i,j,k-1,4) );

    q =  kappa*( dTdx * b.ksx(i,j,k) +
                 dTdy * b.ksy(i,j,k) +
                 dTdz * b.ksz(i,j,k) );

    // flow work
    // Compute face normal volume flux vector
    uf = 0.5*(b.q(i,j,k,1)+b.q(i,j,k-1,1));
    vf = 0.5*(b.q(i,j,k,2)+b.q(i,j,k-1,2));
    wf = 0.5*(b.q(i,j,k,3)+b.q(i,j,k-1,3));

    b.kF(i,j,k,4) = ( uf*txx*b.ksx(i,j,k) +
                      vf*txy*b.ksy(i,j,k) +
                      wf*txz*b.ksz(i,j,k) ) - q;

    // Species
    double Dij;
    for (int n=0; n<th.ns-1; n++)
    {
      Dij  = 0.5 * ( b.qt(i,j,k,2+n) + b.qt(i,j,k-1,2+n)     );
      dNdx = 0.5 * ( b.dqdx(i,j,k,5+n) + b.dqdx(i,j,k-1,5+n) );
      dNdy = 0.5 * ( b.dqdy(i,j,k,5+n) + b.dqdy(i,j,k-1,5+n) );
      dNdz = 0.5 * ( b.dqdz(i,j,k,5+n) + b.dqdz(i,j,k-1,5+n) );

      b.kF(i,j,k,5+n) = -Dij*( dNdx * b.ksx(i,j,k) +
                               dNdy * b.ksy(i,j,k) +
                               dNdz * b.ksz(i,j,k) );
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

};