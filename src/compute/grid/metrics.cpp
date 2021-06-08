
#include "Kokkos_Core.hpp"
#include "kokkos_types.hpp"
#include "block_.hpp"

void metrics(block_ b) {

    int ngls = b.ngls;

    MDRange3 range({{0,0,0}},{{b.ni,b.nj,b.nk}});
    Kokkos::parallel_for("add3", range, KOKKOS_LAMBDA(const int i,
                                                      const int j,
                                                      const int k) {

      b.x_(i,j,k) += 0.0;
      b.y_(i,j,k) += 0.0;
      b.z_(i,j,k) += 0.0;

  });
}
