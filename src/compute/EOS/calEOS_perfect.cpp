#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include <math.h>

void calEOS_perfect(block_ b,
              std::string face="0",
              std::string given="PT") {

  MDRange3 range = get_range3(b, face);

  double cp = 1006.0;

  if ( given.compare("PT") == 0 )
  {
  Kokkos::parallel_for("Compute total energy from primatives",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  b.Q(i,j,k,4) = cp*b.q(i,j,k,4) + 0.5*sqrt(pow(b.q(i,j,k,1),2.0) +
                                            pow(b.q(i,j,k,2),2.0) +
                                            pow(b.q(i,j,k,3),2.0) );
  });
  }
  else if ( given.compare("Erho") == 0 )
  {
  Kokkos::parallel_for("Compute total energy from primatives",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  b.q(i,j,k,4) = ( b.Q(i,j,k,4) + 0.5*sqrt(pow(b.q(i,j,k,1),2.0) +
                                           pow(b.q(i,j,k,2),2.0) +
                                           pow(b.q(i,j,k,3),2.0) ) )/cp;
  });
  }
}
