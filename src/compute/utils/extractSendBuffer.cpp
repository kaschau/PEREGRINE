#include "faceBuffers.hpp"
#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

PG_ABI void pgExtractSendBuffer(const pgView *view_, const pgView *buffer_,
                                int nface, const int *slices, int nLayer,
                                int transpose, int flip0, int flip1) {
  if (view_->rank == 4)
    pack(as4(*view_), as4(*buffer_), nface, slices, nLayer, transpose, flip0,
         flip1);
  else
    pack(as5(*view_), as5(*buffer_), nface, slices, nLayer, transpose, flip0,
         flip1);
}
