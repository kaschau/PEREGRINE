#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include <math.h>
#include <iostream>

void EOS_ideal(block_ b,
               std::string face="0",
               std::string given="PT") {

  MDRange3 range = get_range3(b, face);

  double R = 281.4583333333333;

  if ( given.compare("PT") == 0 )
  {
  Kokkos::parallel_for("Compute total energy from primatives",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  b.Q(i,j,k,0) = b.q(i,j,k,0)/(R*b.q(i,j,k,4));

  });
  }
}
