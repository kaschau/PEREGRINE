#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"

void dq4FD(block_ b) {

//-------------------------------------------------------------------------------------------|
// Spatial derivatices of primative variables
// estimated via fourth order finite difference
//-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng-1, b.ng-1, b.ng-1,0},{b.ni+b.ng,
                                                b.nj+b.ng,
                                                b.nk+b.ng,
                                                b.ne});
  Kokkos::parallel_for("4th order spatial deriv",
                       range_cc,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k,
                                     const int l) {

    double term1, term2, term3;

    term1 = -b.q(i+2, j  , k  , l) + 8.0*b.q(i+1, j  , k  , l) - 8.0*b.q(i-1, j  , k  , l) + b.q(i+2, j  , k  , l) ;
    term2 = -b.q(i  , j+2, k  , l) + 8.0*b.q(i  , j+1, k  , l) - 8.0*b.q(i  , j-1, k  , l) + b.q(i  , j+2, k  , l) ;
    term3 = -b.q(i  , j  , k+2, l) + 8.0*b.q(i  , j  , k+1, l) - 8.0*b.q(i  , j  , k-1, l) + b.q(i  , j  , k+2, l) ;

    b.dqdx(i,j,k,l) = ( term1*b.dEdx(i,j,k) + term2*b.dNdx(i,j,k) + term3*b.dXdx(i,j,k) ) / 12.0;
    b.dqdy(i,j,k,l) = ( term1*b.dEdy(i,j,k) + term2*b.dNdy(i,j,k) + term3*b.dXdy(i,j,k) ) / 12.0;
    b.dqdz(i,j,k,l) = ( term1*b.dEdz(i,j,k) + term2*b.dNdz(i,j,k) + term3*b.dXdz(i,j,k) ) / 12.0;

  });

}
