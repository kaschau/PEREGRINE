#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "thtrdat_.hpp"

void fourthOrderKEEP(block_ b, const thtrdat_ th, const double primary) {

//-------------------------------------------------------------------------------------------|
// i flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_i({b.ng,b.ng,b.ng},{b.ni+b.ng, b.nj+b.ng-1, b.nk+b.ng-1});
  Kokkos::parallel_for("i face conv fluxes", range_i, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {

    const int order = 4;
    const std::array<double, 2> aq = {2.0/3.0, -1.0/12.0};

    double U;
    double uf=0.0;
    double vf=0.0;
    double wf=0.0;

    double a;
    // Compute face normal volume flux vector
    double tempu=0.0;
    double tempv=0.0;
    double tempw=0.0;
    for (int is=1; is <= order/2; is++) {
      a = aq[is-1];
      tempu = 0.0;
      tempv = 0.0;
      tempw = 0.0;
      for (int js=0; js <= is-1; js++) {
        tempu += 0.5*( b.q(i+js,j,k,1) + b.q(i+js-is,j,k,1) );
        tempv += 0.5*( b.q(i+js,j,k,2) + b.q(i+js-is,j,k,2) );
        tempw += 0.5*( b.q(i+js,j,k,3) + b.q(i+js-is,j,k,3) );
      }
      uf += a*tempu;
      vf += a*tempv;
      wf += a*tempw;
    }
    uf *=2.0;
    vf *=2.0;
    wf *=2.0;

    U = b.isx(i,j,k)*uf +
        b.isy(i,j,k)*vf +
        b.isz(i,j,k)*wf ;

    //Compute fluxes

    // Continuity rho*Ui
    double rho = 0.0;
    double temprho = 0.0;
    for (int is=1; is <= order/2; is++) {
      a = aq[is-1];
      temprho = 0.0;
      for (int js=0; js <= is-1; js++) {
        temprho += 0.5*( b.Q(i+js,j,k,0) + b.Q(i+js-is,j,k,0) );
      }
      rho += a*temprho;
    }
    rho *=2.0;

    b.iF(i,j,k,0) = rho * U;

    // x momentum rho*u*Ui+ p*Ax
    // y momentum rho*v*Ui+ p*Ay
    // w momentum rho*w*Ui+ p*Az
    double rhou=0.0;
    double rhov=0.0;
    double rhow=0.0;
    double p=0.0;
    double temprhou, temprhov, temprhow, tempp;
    for (int is=1; is <= order/2; is++) {
      a = aq[is-1];
      temprhou = 0.0;
      temprhov = 0.0;
      temprhow = 0.0;
      tempp = 0.0;
      for (int js=0; js <= is-1; js++) {
        temprhou += 0.5*( b.Q(i+js,j,k,0) + b.Q(i+js-is,j,k,0) ) * ( b.q(i+js,j,k,1) + b.q(i+js-is,j,k,1) );
        temprhov += 0.5*( b.Q(i+js,j,k,0) + b.Q(i+js-is,j,k,0) ) * ( b.q(i+js,j,k,2) + b.q(i+js-is,j,k,2) );
        temprhow += 0.5*( b.Q(i+js,j,k,0) + b.Q(i+js-is,j,k,0) ) * ( b.q(i+js,j,k,3) + b.q(i+js-is,j,k,3) );
        tempp    += 0.5*( b.q(i+js,j,k,0) + b.q(i+js-is,j,k,0) );
      }
      rhou += a*temprhou;
      rhov += a*temprhov;
      rhow += a*temprhow;
      p += a*tempp;
    }
    rhow *=2.0;
    rhov *=2.0;
    rhou *=2.0;
    p *=2.0;

    b.iF(i,j,k,1) = rhou * U + p*b.isx(i,j,k) ;

    b.iF(i,j,k,2) = rhov * U + p*b.isy(i,j,k) ;

    b.iF(i,j,k,3) = rhow * U + p*b.isz(i,j,k) ;

    // Total energy (rhoE+ p)*Ui)
    double rhoE=0.0;
    double pu=0.0;
    double temprhoE, temppu;
    for (int is=1; is <= order/2; is++) {
      a = aq[is-1];
      temprhoE = 0.0;
      temppu = 0.0;
      for (int js=0; js <= is-1; js++) {
        temprhoE += 0.5* ( b.Q(i+js,j,k,0) + b.Q(i+js-is,j,k,0) )
                       * (
                           (  b.qh(i+js,j,k,4)/b.Q(i,j,k,0)
                            + b.qh(i+js-is,j,k,4)/b.Q(i+js-is,j,k,0) )
                         + (  b.q(i+js,j,k,1)*b.q(i+js-is,j,k,1)
                            + b.q(i+js,j,k,2)*b.q(i+js-is,j,k,2)
                            + b.q(i+js,j,k,3)*b.q(i+js-is,j,k,3) )
                         );

        temppu += 0.5*(
                        b.q(i+js-is,j,k,0)*(b.q(i+js   ,j,k,1)*b.isx(i,j,k)
                                           +b.q(i+js   ,j,k,2)*b.isy(i,j,k)
                                           +b.q(i+js   ,j,k,3)*b.isz(i,j,k) ) +
                        b.q(i+js   ,j,k,0)*(b.q(i+js-is,j,k,1)*b.isx(i,j,k)
                                           +b.q(i+js-is,j,k,2)*b.isy(i,j,k)
                                           +b.q(i+js-is,j,k,3)*b.isz(i,j,k) )
                       );
      }
      rhoE += a*temprhoE;
      pu   += a*temppu;
    }
    rhoE *=2.0;
    pu   *=2.0;

    b.iF(i,j,k,4) = rhoE * U + pu;

    // Species
    for (int n=0; n<th.ns-1; n++)
    {

      double rhoY = 0.0;
      double temprhoY = 0.0;
      for (int is=1; is <= order/2; is++) {
        a = aq[is-1];
        temprhoY = 0.0;
        for (int js=0; js <= is-1; js++) {
          temprhoY += 0.5*( (b.Q(i+js,j,k,0) + b.Q(i+js-is,j,k,0) )*(b.q(i+js,j,k,5+n)+b.q(i+js-is,j,k,5+n) ));
        }
        rhoY += a*temprhoY;
      }
      rhoY *=2.0;
      b.iF(i,j,k,5+n) = rhoY * U;
    }

  });

//-------------------------------------------------------------------------------------------|
// j flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_j({b.ng,b.ng,b.ng},{b.ni+b.ng-1, b.nj+b.ng, b.nk+b.ng-1});
  Kokkos::parallel_for("j face conv fluxes", range_j, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {


  });

//-------------------------------------------------------------------------------------------|
// k flux face range
//-------------------------------------------------------------------------------------------|
  MDRange3 range_k({b.ng,b.ng,b.ng},{b.ni+b.ng-1, b.nj+b.ng-1, b.nk+b.ng});
  Kokkos::parallel_for("k face conv fluxes", range_k, KOKKOS_LAMBDA(const int i,
                                                                    const int j,
                                                                    const int k) {


  });


//-------------------------------------------------------------------------------------------|
// Apply fluxes to cc range
//-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng,b.ng,b.ng,0},{b.ni+b.ng-1,b.nj+b.ng-1,b.nk+b.ng-1,b.ne});
  Kokkos::parallel_for("Apply current fluxes to RHS",
                       range_cc,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k,
                                     const int l) {


    // Compute switch on face
    double iFphi  = std::max( b.phi(i,j,k,0) , b.phi(i-1,j,k,0) );
    double iFphi1 = std::max( b.phi(i,j,k,0) , b.phi(i+1,j,k,0) );
    double jFphi  = std::max( b.phi(i,j,k,1) , b.phi(i,j-1,k,1) );
    double jFphi1 = std::max( b.phi(i,j,k,1) , b.phi(i,j+1,k,1) );
    double kFphi  = std::max( b.phi(i,j,k,2) , b.phi(i,j,k-1,2) );
    double kFphi1 = std::max( b.phi(i,j,k,2) , b.phi(i,j,k+1,2) );

    const double dPrimary = 2.0*primary - 1.0;

    // Add fluxes to RHS
    // format is F_primary*(1-switch) + F_secondary*(switch)
    b.dQ(i,j,k,l) += ( b.iF(i,j,k,l) * (primary -  iFphi * dPrimary) +
                       b.jF(i,j,k,l) * (primary -  jFphi * dPrimary) +
                       b.kF(i,j,k,l) * (primary -  kFphi * dPrimary) ) / b.J(i,j,k) ;

    b.dQ(i,j,k,l) -= ( b.iF(i+1,j,k,l) * (primary -  iFphi1 * dPrimary) +
                       b.jF(i,j+1,k,l) * (primary -  jFphi1 * dPrimary) +
                       b.kF(i,j,k+1,l) * (primary -  kFphi1 * dPrimary) ) / b.J(i,j,k) ;

  });

};
