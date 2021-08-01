#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include "compute.hpp"
#include <string>
#include <iostream>

void species(block_ b,
              std::string face="0",
              std::string given="rhoY") {

  MDRange3 range = get_range3(b, face);

  if ( given.compare("rhoY") == 0 )
  {
  Kokkos::parallel_for("Compute u,v,w from momentum",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  for (int n=1; n<b.ns; n++)
  {
    b.q(i,j,k,5+n) = b.Q(i,j,k,5+n) / b.Q(i,j,k,0);
  }
  });
  }
  else if ( given.compare("Y") == 0 )
  {
  Kokkos::parallel_for("Compute momentum from primatives",
                       range,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {
  for (int n=1; n<b.ns; n++)
  {
    b.Q(i,j,k,5+n) = b.q(i,j,k,5+n) * b.Q(i,j,k,0);
  }
  });
  }

}
