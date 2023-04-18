#include "block_.hpp"
#include "compute.hpp"
#include "face_.hpp"
#include "kokkosTypes.hpp"

void extractSendBuffer(threeDview &view, threeDview &buffer, face_ &face,
                       const std::vector<int> &slices) {

  int &nface = face._nface;
  int nLayer = slices.size();

  for (int g = 0; g < nLayer; g++) {
    int s = slices[g];

    twoDsubview viewSlice = getFaceSlice(view, nface, s);
    twoDsubview bufferSlice =
        Kokkos::subview(buffer, g, Kokkos::ALL, Kokkos::ALL);

    Kokkos::deep_copy(bufferSlice, viewSlice);
  }
}

void extractSendBuffer(fourDview &view, fourDview &buffer, face_ &face,
                       const std::vector<int> &slices) {

  int &nface = face._nface;
  int nLayer = slices.size();

  for (int g = 0; g < nLayer; g++) {
    int s = slices[g];

    threeDsubview viewSlice = getFaceSlice(view, nface, s);
    threeDsubview bufferSlice =
        Kokkos::subview(buffer, g, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    Kokkos::deep_copy(bufferSlice, viewSlice);
  }
}

void placeRecvBuffer(threeDview &view, threeDview &buffer, face_ &face,
                     const std::vector<int> &slices) {

  int &nface = face._nface;
  int nLayer = slices.size();

  for (int g = 0; g < nLayer; g++) {
    int s = slices[g];

    twoDsubview viewSlice = getFaceSlice(view, nface, s);
    twoDsubview bufferSlice =
        Kokkos::subview(buffer, g, Kokkos::ALL, Kokkos::ALL);

    Kokkos::deep_copy(viewSlice, bufferSlice);
  }
}

void placeRecvBuffer(fourDview &view, fourDview &buffer, face_ &face,
                     const std::vector<int> &slices) {

  int &nface = face._nface;
  int nLayer = slices.size();

  for (int g = 0; g < nLayer; g++) {
    int s = slices[g];

    threeDsubview viewSlice = getFaceSlice(view, nface, s);
    threeDsubview bufferSlice =
        Kokkos::subview(buffer, g, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    Kokkos::deep_copy(viewSlice, bufferSlice);
  }
}
