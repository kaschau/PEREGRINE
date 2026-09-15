#include "faceBuffers.hpp"

// Every trading face's halo, from what its neighbor packed, in one launch.
PG_RANGE(trades)
PG_ABI void pgPlaceRecvBuffer(const recvTrade &k, const pgTiling &t,
                              const int *nface) {
  if (!t.tiles)
    return;
  if (k.ndim == 4)
    forTrades("place recv buffers", t, unpacking<4>{k.view, k.buffer}, nface);
  else
    forTrades("place recv buffers", t, unpacking<5>{k.view, k.buffer}, nface);
}
