#include "faceBuffers.hpp"
#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

// Every trading face's planes, packed for its neighbor, in one call.
PG_ABI void pgExtractSendBuffer(int count, pgIn *view_, pgOut *buffer_,
                                const int *nface, const int *nLayer,
                                const int *skip, const int *transpose,
                                const int *flip0, const int *flip1) {
  for (int e = 0; e < count; e++) {
    if (view_[e].rank == 4)
      pack(as4(view_[e]), as4(buffer_[e]), nface[e], nLayer[e], skip[e],
           transpose[e], flip0[e], flip1[e]);
    else
      pack(as5(view_[e]), as5(buffer_[e]), nface[e], nLayer[e], skip[e],
           transpose[e], flip0[e], flip1[e]);
  }
}
