#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"

void dQzero(block_ b) {

//-------------------------------------------------------------------------------------------|
// Zero out dQ
//-------------------------------------------------------------------------------------------|
  MDRange4 range_cc({b.ng, b.ng, b.ng,0},{b.ni+b.ng-1,
                                          b.nj+b.ng-1,
                                          b.nk+b.ng-1,
                                          b.ne});
  Kokkos::parallel_for("Zero out dQ",
                       range_cc,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k,
                                     const int l) {

    b.dQ(i,j,k,l) = 0.0;

  });

}
