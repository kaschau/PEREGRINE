#include "block_.hpp"
#include "compute.hpp"
#include "face_.hpp"
#include "kokkosTypes.hpp"

void extractSendBuffer(fourDview &view, fourDview &buffer, face_ &face,
                       const std::vector<int> &slices) {

  int &nface = face.nface;
  int nLayer = slices.size();
  // the buffer is packed the way our neighbor reads it, so the plane is
  // turned on the way out rather than after it lands
  const bool transpose = face.orientTranspose;
  const bool flip0 = face.orientFlip0;
  const bool flip1 = face.orientFlip1;

  for (int g = 0; g < nLayer; g++) {
    int s = slices[g];

    threeDsubview viewSlice = getFaceSlice(view, nface, s);
    threeDsubview bufferSlice =
        Kokkos::subview(buffer, g, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    const int na = bufferSlice.extent(0);
    const int nb = bufferSlice.extent(1);
    MDRange3 range({0, 0, 0}, {static_cast<long>(na), static_cast<long>(nb),
                               static_cast<long>(bufferSlice.extent(2))});
    Kokkos::parallel_for(
        "extract send buffer", range,
        KOKKOS_LAMBDA(const int a, const int b, const int l) {
          const int aa = flip0 ? na - 1 - a : a;
          const int bb = flip1 ? nb - 1 - b : b;
          bufferSlice(a, b, l) =
              transpose ? viewSlice(bb, aa, l) : viewSlice(aa, bb, l);
        });
  }
}

void extractSendBuffer(fiveDview &view, fiveDview &buffer, face_ &face,
                       const std::vector<int> &slices) {

  int &nface = face.nface;
  int nLayer = slices.size();
  const bool transpose = face.orientTranspose;
  const bool flip0 = face.orientFlip0;
  const bool flip1 = face.orientFlip1;

  for (int g = 0; g < nLayer; g++) {
    int s = slices[g];

    fourDsubview viewSlice = getFaceSlice(view, nface, s);
    fourDsubview bufferSlice = Kokkos::subview(
        buffer, g, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    const int na = bufferSlice.extent(0);
    const int nb = bufferSlice.extent(1);
    MDRange4 range({0, 0, 0, 0}, {static_cast<long>(na), static_cast<long>(nb),
                                  static_cast<long>(bufferSlice.extent(2)),
                                  static_cast<long>(bufferSlice.extent(3))});
    Kokkos::parallel_for(
        "extract send buffer", range,
        KOKKOS_LAMBDA(const int a, const int b, const int l, const int d) {
          const int aa = flip0 ? na - 1 - a : a;
          const int bb = flip1 ? nb - 1 - b : b;
          bufferSlice(a, b, l, d) =
              transpose ? viewSlice(bb, aa, l, d) : viewSlice(aa, bb, l, d);
        });
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
