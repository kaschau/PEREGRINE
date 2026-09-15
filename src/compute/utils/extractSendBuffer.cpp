#include "faceBuffers.hpp"

// Every trading face's planes, packed for its neighbor, in one launch.
PG_RANGE(bufferPlanes)
PG_ABI void pgExtractSendBuffer(const haloSend &k, const pgTiling &t,
                                const int *nface) {
  if (!t.tiles)
    return;
  if (k.ndim == 4)
    forHaloExchange(
        "extract send buffers", t,
        packing<4>{k.view, k.buffer, k.skip, k.transpose, k.flip0, k.flip1},
        nface);
  else
    forHaloExchange(
        "extract send buffers", t,
        packing<5>{k.view, k.buffer, k.skip, k.transpose, k.flip0, k.flip1},
        nface);
}
