#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"

void applyFlux(block_ b, const double primary) {

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

    // Add fluxes to RHS
    b.dQ(i,j,k,l) += ( b.iF(i  ,j  ,k  ,l) +
                       b.jF(i  ,j  ,k  ,l) +
                       b.kF(i  ,j  ,k  ,l) ) / b.J(i,j,k) ;

    b.dQ(i,j,k,l) -= ( b.iF(i+1,j  ,k  ,l) +
                       b.jF(i  ,j+1,k  ,l) +
                       b.kF(i  ,j  ,k+1,l) ) / b.J(i,j,k) ;

  });

};


void applyHybridFlux(block_ b, const double primary) {

//-------------------------------------------------------------------------------------------|
// Apply fluxes to cc range
//-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng,b.ng,b.ng,0},{b.ni+b.ng-1,b.nj+b.ng-1,b.nk+b.ng-1,b.ne});
  Kokkos::parallel_for("Apply hybrid fluxes to RHS",
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

    double dPrimary = 2.0*primary - 1.0;

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

void applyDissipationFlux(block_ b, const double primary) {

//-------------------------------------------------------------------------------------------|
// Apply fluxes to cc range
//-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng,b.ng,b.ng,0},{b.ni+b.ng-1,b.nj+b.ng-1,b.nk+b.ng-1,b.ne});
  Kokkos::parallel_for("Apply dissipaiton fluxes to RHS",
                       range_cc,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k,
                                     const int l) {

    // Add fluxes to RHS
    b.dQ(i,j,k,l) -= ( b.iF(i  ,j  ,k  ,l) +
                       b.jF(i  ,j  ,k  ,l) +
                       b.kF(i  ,j  ,k  ,l) ) / b.J(i,j,k) ;

    b.dQ(i,j,k,l) += ( b.iF(i+1,j  ,k  ,l) +
                       b.jF(i  ,j+1,k  ,l) +
                       b.kF(i  ,j  ,k+1,l) ) / b.J(i,j,k) ;

  });

};
