
#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"

void metrics(block_ b) {

    MDRange3 range({{0,0,0}},{{b.ni+1,b.nj+1,b.nk+1}});
    Kokkos::parallel_for("add3", range, KOKKOS_LAMBDA(const int i,
                                                      const int j,
                                                      const int k) {

      b.xc(i,j,k) = 1.0;
      b.yc(i,j,k) = 1.0;
      b.zc(i,j,k) = 1.0;

  });
}
