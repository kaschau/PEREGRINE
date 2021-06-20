#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include <string>
#include <iostream>

void momentum(block_ b,
              std::string face="0",
              std::string given="rhou") {

  MDRange3 range = get_range3(b, face);

  if ( given.compare("rhou") == 0 )
  {
  Kokkos::parallel_for("Compute total energy from primatives",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  b.q(i,j,k,1) = b.Q(i,j,k,1) / b.Q(i,j,k,0);
  b.q(i,j,k,2) = b.Q(i,j,k,2) / b.Q(i,j,k,0);
  b.q(i,j,k,3) = b.Q(i,j,k,3) / b.Q(i,j,k,0);
  });
  }
  else if ( given.compare("u") == 0 )
  {
  Kokkos::parallel_for("Compute total energy from primatives",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {
  b.Q(i,j,k,1) = b.q(i,j,k,1) * b.Q(i,j,k,0);
  b.Q(i,j,k,2) = b.q(i,j,k,2) * b.Q(i,j,k,0);
  b.Q(i,j,k,3) = b.q(i,j,k,3) * b.Q(i,j,k,0);
  });
  }

}
