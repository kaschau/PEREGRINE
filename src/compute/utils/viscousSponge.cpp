#include "Kokkos_Core.hpp"
#include "block_.hpp"
#include "kokkosTypes.hpp"

void viscousSponge(block_ &b, const std::vector<double> &origin,
                   const std::vector<double> &ending, double mult) {

  MDRange3 range_cc({b.ng, b.ng, b.ng},
                    {b.ni + b.ng - 1, b.nj + b.ng - 1, b.nk + b.ng - 1});
  Kokkos::parallel_for(
      "Apply viscous sponge", range_cc,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        double &xc = b.xc(i, j, k);
        double &yc = b.yc(i, j, k);
        double &zc = b.zc(i, j, k);

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
        multiplier = 1.0 + fmax(0.0, multiplier);

        b.qt(i, j, k, 0) *= multiplier;
      });
}