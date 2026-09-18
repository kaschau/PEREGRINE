// Jameson's pressure sensor: the second difference of the pressure along
// the face normal over its sum, at the cell either side of the face, the
// larger, times the gain and clipped to one -- the raw ratio peaks near a
// third across a shock. The gain is the config's switchValues, baked in.
// The cells beyond the two are read: PG_STENCIL(2).
#ifndef __switchJamesonPressure_H__
#define __switchJamesonPressure_H__

#include "faces.hpp"

#ifndef PG_JAMESONPRESSURE_GAIN
#error "jamesonPressure takes switchValues gain"
#endif

PG_STENCIL(2);

struct jamesonPressure : PG_RECONSTRUCT {
  using base = PG_RECONSTRUCT;
  static KOKKOS_INLINE_FUNCTION fpdtype sensor(fpdtype pm, fpdtype p,
                                               fpdtype pp) {
    return fabs(pp - 2.0 * p + pm) / fabs(pp + 2.0 * p + pm);
  }
  KOKKOS_INLINE_FUNCTION fpdtype weight() const {
    const fpdtype pLL = this->qL(-N, 0), pL = this->qL(0), pR = this->qR(0),
                  pRR = this->qR(+N, 0);
    const fpdtype s = fmax(sensor(pLL, pL, pR), sensor(pL, pR, pRR));
    return fmin(PG_JAMESONPRESSURE_GAIN * s, 1.0);
  }
};

#endif
