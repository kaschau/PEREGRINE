#include "faceBuffers.hpp"

// Every trading face's planes, packed for its neighbor, in one launch.
PG_RANGE(trades)
PG_ABI void pgExtractSendBuffer(const sendTrade &k, const pgTiling &t,
                                const int *nface) {
  if (!t.tiles)
    return;
  if (k.ndim == 4)
    forTrades(
        "extract send buffers", t,
        packing<4>{k.view, k.buffer, k.skip, k.transpose, k.flip0, k.flip1},
        nface);
  else
    forTrades(
        "extract send buffers", t,
        packing<5>{k.view, k.buffer, k.skip, k.transpose, k.flip0, k.flip1},
        nface);
}
