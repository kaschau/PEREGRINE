#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

PG_ABI void pgViscousSponge(int count, pgIn *cells_, pgOut *qt_,
                            const pgDims *d, const double *origin,
                            const double *ending, double mult) {
  for (int e = 0; e < count; e++) {
    auto cells = as4(cells_[e]);
    auto qt = as4(qt_[e]);
    const int ni = d[e].ni, nj = d[e].nj, nk = d[e].nk;

    MDRange3 range_cc({ng - 1, ng - 1, ng - 1}, {ni + ng, nj + ng, nk + ng});
    Kokkos::parallel_for(
        "Apply viscous sponge", range_cc,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          const double &xc = cells(i, j, k, 0);
          const double &yc = cells(i, j, k, 1);
          const double &zc = cells(i, j, k, 2);

          double vectorX = xc - origin[0];
          double vectorY = yc - origin[1];
          double vectorZ = zc - origin[2];

          double spongeLength = sqrt(pow(ending[0] - origin[0], 2.0) +
                                     pow(ending[1] - origin[1], 2.0) +
                                     pow(ending[2] - origin[2], 2.0));

          double normal[3] = {
              (ending[0] - origin[0]) / spongeLength,
              (ending[1] - origin[1]) / spongeLength,
              (ending[2] - origin[2]) / spongeLength,
          };

          double dist =
              vectorX * normal[0] + vectorY * normal[1] + vectorZ * normal[2];

          double multiplier = dist / spongeLength * (mult - 1.0);
          multiplier = 1.0 + fmin(fmax(0.0, multiplier), mult);

          qt(i, j, k, 0) *= multiplier;
        });
  }
}
