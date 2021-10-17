#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"

void noIFlux(block_ b) {

  MDRange3 range_cc({0,0,0},{b.ni+2*b.ng-1,b.nj+2*b.ng-1,b.nk+2*b.ng-1});

  Kokkos::parallel_for("Compute switch from pressure",
                       range_cc,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  b.phi(i,j,k,0) = 1.0 ;
  b.phi(i,j,k,1) = 0.0 ;
  b.phi(i,j,k,2) = 0.0 ;

  });
}

void noJFlux(block_ b) {

  MDRange3 range_cc({0,0,0},{b.ni+2*b.ng-1,b.nj+2*b.ng-1,b.nk+2*b.ng-1});

  Kokkos::parallel_for("Compute switch from pressure",
                       range_cc,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  b.phi(i,j,k,0) = 0.0 ;
  b.phi(i,j,k,1) = 1.0 ;
  b.phi(i,j,k,2) = 0.0 ;

  });
}

void noKFlux(block_ b) {

  MDRange3 range_cc({0,0,0},{b.ni+2*b.ng-1,b.nj+2*b.ng-1,b.nk+2*b.ng-1});

  Kokkos::parallel_for("Compute switch from pressure",
                       range_cc,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  b.phi(i,j,k,0) = 0.0 ;
  b.phi(i,j,k,1) = 0.0 ;
  b.phi(i,j,k,2) = 1.0 ;

  });
}

void noInoJFlux(block_ b) {

  MDRange3 range_cc({0,0,0},{b.ni+2*b.ng-1,b.nj+2*b.ng-1,b.nk+2*b.ng-1});

  Kokkos::parallel_for("Compute switch from pressure",
                       range_cc,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  b.phi(i,j,k,0) = 1.0 ;
  b.phi(i,j,k,1) = 1.0 ;
  b.phi(i,j,k,2) = 0.0 ;

  });
}

void noInoKFlux(block_ b) {

  MDRange3 range_cc({0,0,0},{b.ni+2*b.ng-1,b.nj+2*b.ng-1,b.nk+2*b.ng-1});

  Kokkos::parallel_for("Compute switch from pressure",
                       range_cc,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  b.phi(i,j,k,0) = 1.0 ;
  b.phi(i,j,k,1) = 0.0 ;
  b.phi(i,j,k,2) = 1.0 ;

  });
}

void noJnoKFlux(block_ b) {

  MDRange3 range_cc({0,0,0},{b.ni+2*b.ng-1,b.nj+2*b.ng-1,b.nk+2*b.ng-1});

  Kokkos::parallel_for("Compute switch from pressure",
                       range_cc,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  b.phi(i,j,k,0) = 0.0 ;
  b.phi(i,j,k,1) = 1.0 ;
  b.phi(i,j,k,2) = 1.0 ;

  });
}
