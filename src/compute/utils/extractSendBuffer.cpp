#include "faceBuffers.hpp"
#include "kernelUtils.hpp"
#include "kokkosTypes.hpp"
#include <Kokkos_Core.hpp>

PG_ABI void pgExtractSendBuffer(pgIn *view_, pgOut *buffer_, int nface,
                                int nLayer, int skip, int transpose, int flip0,
                                int flip1) {
  if (view_->rank == 4)
    pack(as4(*view_), as4(*buffer_), nface, nLayer, skip, transpose, flip0,
         flip1);
  else
    pack(as5(*view_), as5(*buffer_), nface, nLayer, skip, transpose, flip0,
         flip1);
}
