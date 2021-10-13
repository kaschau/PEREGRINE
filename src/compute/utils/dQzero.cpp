#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"

void dQzero(block_ b) {

//-------------------------------------------------------------------------------------------|
// Zero out dQ
//-------------------------------------------------------------------------------------------|
  MDRange4 range({b.ng,b.ng,b.ng,0},{b.ni+2*b.ng-1,b.nj+2*b.ng-1,b.nk+2*b.ng-1,b.ne});
  Kokkos::parallel_for("Apply current fluxes to RHS",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k,
                                     const int l) {

    b.dQ(i,j,k,l) = 0.0;

  });

};
