#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"
#include <vector>
#include <math.h>

void total_energy(std::vector<block_> mb) {

for(block_ b : mb){
  // cc range
  MDRange3 range_i({0,0,0},{b.ni+1,b.nj+1,b.nk+1});
  Kokkos::parallel_for("Compute total energy from primatives",
                       range_i,
                       KOKKOS_LAMBDA(const int i,
                                     const int j,
                                     const int k) {

  double cp = 1006.0;

  b.Q(i,j,k,4) = cp*b.q(i,j,k,4) + 0.5*b.Q(i,j,k,0)*sqrt(pow(b.q(i,j,k,1),2.0) +
                                                         pow(b.q(i,j,k,2),2.0) +
                                                         pow(b.q(i,j,k,3),2.0) );

  });

}};
