#include "kernel.hpp"

PG_RANGE(interiorPlusOne)
struct viscousSponge {
  in cells;
  inout qt;
  dims d;
  const double *origin, *ending;
  double mult;
  KOKKOS_INLINE_FUNCTION void operator()() const {
    const int ni = d->ni, nj = d->nj, nk = d->nk;

    const double &xc = cells(0);
    const double &yc = cells(1);
    const double &zc = cells(2);

    double vectorX = xc - origin[0];
    double vectorY = yc - origin[1];
    double vectorZ = zc - origin[2];

    double spongeLength =
        sqrt(pow(ending[0] - origin[0], 2.0) + pow(ending[1] - origin[1], 2.0) +
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

    qt(0) *= multiplier;
  }
};

PG_ABI void pgViscousSponge(const viscousSponge &k, const pgTiling &t) {
  forCells("Apply viscous sponge", t, k);
}
