#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include <math.h>
#include <iostream>
#include <stdexcept>

void EOS_ideal(block_ b,
               std::string face,
               std::string given) {

  MDRange3 range = get_range3(b, face);

  double R = 281.4583333333333;

  if ( given.compare("PT") == 0 )
  {
  Kokkos::parallel_for("Compute density from P and T",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  b.Q(i,j,k,0) = b.q(i,j,k,0)/(R*b.q(i,j,k,4));

  });
  }
  else if ( given.compare("rhoT") == 0 )
  {
  Kokkos::parallel_for("Compute pressure from rho and T",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  b.q(i,j,k,0) = b.Q(i,j,k,0)*R*b.q(i,j,k,4);

  });
  }
  else
  {
  throw std::invalid_argument( "Invalid given string in EOSideal.");
  }
}
