#include "faceBuffers.hpp"

// Every face met on this rank: its halo filled directly from the block
// across, in one launch. Direct is only possible within a rank.
PG_RANGE(bufferPlanes)
PG_ABI void pgDirectHaloFill(const directHaloFill &k, const pgTiling &t,
                             const int *nface) {
  if (!t.tiles)
    return;
  if (k.ndim == 4)
    forBufferPlanes("direct halo fill", t,
                    directFilling<4>{k.view, k.theirs, k.partnerNface,
                                     k.partnerTranspose, k.partnerFlip0,
                                     k.partnerFlip1, k.skip},
                    nface);
  else
    forBufferPlanes("direct halo fill", t,
                    directFilling<5>{k.view, k.theirs, k.partnerNface,
                                     k.partnerTranspose, k.partnerFlip0,
                                     k.partnerFlip1, k.skip},
                    nface);
}
