
#include "kokkos2peregrine.hpp"
#include "block.hpp"
#include "Kokkos_Core.hpp"
#include <cstdint>

void add3(block b, double n ) {

  MDRange3 _range({{0,0,0}},{{b.ni,b.nj,b.nk}});
  Kokkos::parallel_for("add3", _range, KOKKOS_LAMBDA(const int i, const int j, const int k) {

      b.x(i,j,k) += n;
      b.y(i,j,k) += n;
      b.z(i,j,k) += n;

  });
}

