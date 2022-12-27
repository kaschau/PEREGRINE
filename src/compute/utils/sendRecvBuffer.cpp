#include "block_.hpp"
#include "compute.hpp"
#include "face_.hpp"
#include "kokkosTypes.hpp"

void extract_sendBuffer3(threeDview &view, face_ &face,
                         const std::vector<int> &slices) {

  int &nface = face._nface;
  int nLayer = slices.size();

  threeDview &buffer = face.tempRecvBuffer3;

  for (int g = 0; g < nLayer; g++) {
    int s = slices[g];

    twoDsubview viewSlice = getHaloSlice(view, nface, s);
    twoDsubview bufferSlice =
        Kokkos::subview(buffer, g, Kokkos::ALL, Kokkos::ALL);

    Kokkos::deep_copy(bufferSlice, viewSlice);
  }
}

void extract_sendBuffer4(fourDview &view, face_ &face,
                         const std::vector<int> &slices) {

  int &nface = face._nface;
  int nLayer = slices.size();

  fourDview &buffer = face.tempRecvBuffer4;
  for (int g = 0; g < nLayer; g++) {
    int s = slices[g];

    threeDsubview viewSlice = getHaloSlice(view, nface, s);
    threeDsubview bufferSlice =
        Kokkos::subview(buffer, g, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    Kokkos::deep_copy(bufferSlice, viewSlice);
  }
}

void place_recvBuffer3(threeDview &view, face_ &face,
                       const std::vector<int> &slices) {

  int &nface = face._nface;
  int nLayer = slices.size();

  threeDview &buffer = face.recvBuffer3;

  for (int g = 0; g < nLayer; g++) {
    int s = slices[g];

    twoDsubview viewSlice = getHaloSlice(view, nface, s);
    twoDsubview bufferSlice =
        Kokkos::subview(buffer, g, Kokkos::ALL, Kokkos::ALL);

    Kokkos::deep_copy(viewSlice, bufferSlice);
  }
}

void place_recvBuffer4(fourDview &view, face_ &face,
                       const std::vector<int> &slices) {

  int &nface = face._nface;
  int nLayer = slices.size();

  fourDview &buffer = face.recvBuffer4;

  for (int g = 0; g < nLayer; g++) {
    int s = slices[g];

    threeDsubview viewSlice = getHaloSlice(view, nface, s);
    threeDsubview bufferSlice =
        Kokkos::subview(buffer, g, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    Kokkos::deep_copy(viewSlice, bufferSlice);
  }
}
