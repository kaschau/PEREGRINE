#include "faceBuffers.hpp"

// Every rotational periodic face's halo, turned by its rotation once the
// halo has landed, in one launch.
PG_RANGE(bufferPlanes)
PG_ABI void pgTurnHalo(const haloTurn &k, const pgTiling &t, const int *nface) {
  if (!t.tiles)
    return;
  if (k.ndim == 4)
    forBufferPlanes("turn halos", t, turning<4>{k.view, k.rotation, k.vectors},
                    nface);
  else
    forBufferPlanes("turn halos", t, turning<5>{k.view, k.rotation, k.vectors},
                    nface);
}
