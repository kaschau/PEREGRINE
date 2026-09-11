#include "block_.hpp"
#include "compute.hpp"
#include "face_.hpp"
#include "kokkosTypes.hpp"

void extractSendBuffer(fourDview &view, fourDview &buffer, face_ &face,
                       const std::vector<int> &slices) {

  int &nface = face.nface;
  int nLayer = slices.size();

  for (int g = 0; g < nLayer; g++) {
    int s = slices[g];

    threeDsubview viewSlice = getFaceSlice(view, nface, s);
    threeDsubview bufferSlice =
        Kokkos::subview(buffer, g, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    Kokkos::deep_copy(bufferSlice, viewSlice);
  }
}

void extractSendBuffer(fiveDview &view, fiveDview &buffer, face_ &face,
                       const std::vector<int> &slices) {

  int &nface = face.nface;
  int nLayer = slices.size();

  for (int g = 0; g < nLayer; g++) {
    int s = slices[g];

    fourDsubview viewSlice = getFaceSlice(view, nface, s);
    fourDsubview bufferSlice = Kokkos::subview(
        buffer, g, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    Kokkos::deep_copy(bufferSlice, viewSlice);
  }
}

void placeRecvBuffer(fourDview &view, fourDview &buffer, face_ &face,
                     const std::vector<int> &slices) {

  int &nface = face.nface;
  int nLayer = slices.size();

  for (int g = 0; g < nLayer; g++) {
    int s = slices[g];

    threeDsubview viewSlice = getFaceSlice(view, nface, s);
    threeDsubview bufferSlice =
        Kokkos::subview(buffer, g, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    Kokkos::deep_copy(viewSlice, bufferSlice);
  }
}

void placeRecvBuffer(fiveDview &view, fiveDview &buffer, face_ &face,
                     const std::vector<int> &slices) {

  int &nface = face.nface;
  int nLayer = slices.size();

  for (int g = 0; g < nLayer; g++) {
    int s = slices[g];

    fourDsubview viewSlice = getFaceSlice(view, nface, s);
    fourDsubview bufferSlice = Kokkos::subview(
        buffer, g, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    Kokkos::deep_copy(viewSlice, bufferSlice);
  }
}
