#include "faceBuffers.hpp"
#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

PG_ABI void pgPlaceRecvBuffer(pgOut *view_, pgIn *buffer_, int nface,
                              const int *slices, int nLayer) {
  if (view_->rank == 4)
    unpack(as4(*view_), as4(*buffer_), nface, slices, nLayer);
  else
    unpack(as5(*view_), as5(*buffer_), nface, slices, nLayer);
}
