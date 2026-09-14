#include "faceBuffers.hpp"
#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

// Every trading face's halo, from what its neighbor packed, in one call.
PG_ABI void pgPlaceRecvBuffer(int count, pgOut *view_, pgIn *buffer_,
                              const int *nface, const int *nLayer) {
  for (int e = 0; e < count; e++) {
    if (view_[e].rank == 4)
      unpack(as4(view_[e]), as4(buffer_[e]), nface[e], nLayer[e]);
    else
      unpack(as5(view_[e]), as5(buffer_[e]), nface[e], nLayer[e]);
  }
}
