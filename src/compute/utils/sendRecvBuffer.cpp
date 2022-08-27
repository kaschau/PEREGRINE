#include "block_.hpp"
#include "compute.hpp"
#include "face_.hpp"
#include "kokkos_types.hpp"
#include <Kokkos_CopyViews.hpp>

void extract_sendBuffer3(threeDview &view, face_ &face,
                         const std::vector<int> &slices) {

  int &nface = face._nface;
  int &ng = face._ng;

  threeDview &buffer = face.tempRecvBuffer3;

  // MDRange2 range_face = MDRange2({0, 0}, {buffer.extent(0),
  // buffer.extent(1)});

  for (int g = 0; g < ng; g++) {
    int s = slices[g];

    twoDsubview viewSlice = getHaloSlice(view, nface, s);
    twoDsubview bufferSlice =
        Kokkos::subview(buffer, g, Kokkos::ALL, Kokkos::ALL);

    Kokkos::deep_copy(bufferSlice, viewSlice);

    // Kokkos::parallel_for(
    //     "Copy buffer data", range_face,
    //     KOKKOS_LAMBDA(const int i, const int j) {
    //       // set pressure
    //       bufferSlice(i, j) = viewSlice(i,j)
    //     });
  }
}

void extract_sendBuffer4(fourDview &view, face_ &face,
                         const std::vector<int> &slices) {

  int &nface = face._nface;
  int &ng = face._ng;

  fourDview &buffer = face.tempRecvBuffer4;

  // MDRange3 range_face = MDRange3({0, 0, 0}, {buffer.extent(0),
  // buffer.extent(1), buffer.extent(1)});

  for (int g = 0; g < ng; g++) {
    int s = slices[g];

    threeDsubview viewSlice = getHaloSlice(view, nface, s);
    threeDsubview bufferSlice =
        Kokkos::subview(buffer, g, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    Kokkos::deep_copy(bufferSlice, viewSlice);

    // Kokkos::parallel_for(
    //     "Copy buffer data", range_face,
    //     KOKKOS_LAMBDA(const int i, const int j, const int l) {
    //       // C
    //       bufferSlice(i, j, l) = viewSlice(i, j, l)
    //     });
  }
}

void place_recvBuffer3(threeDview &view, face_ &face,
                       const std::vector<int> &slices) {

  int &nface = face._nface;
  int &ng = face._ng;

  threeDview &buffer = face.recvBuffer3;

  // MDRange2 range_face = MDRange2({0, 0}, {buffer.extent(0),
  // buffer.extent(1)});

  for (int g = 0; g < ng; g++) {
    int s = slices[g];

    twoDsubview viewSlice = getHaloSlice(view, nface, s);
    twoDsubview bufferSlice =
        Kokkos::subview(buffer, g, Kokkos::ALL, Kokkos::ALL);

    Kokkos::deep_copy(viewSlice, bufferSlice);

    // Kokkos::parallel_for(
    //     "Copy buffer data", range_face,
    //     KOKKOS_LAMBDA(const int i, const int j) {
    //       // set pressure
    //       viewSlice(i, j) = bufferSlice(i,j)
    //     });
  }
}

void place_recvBuffer4(fourDview &view, face_ &face,
                       const std::vector<int> &slices) {

  int &nface = face._nface;
  int &ng = face._ng;

  fourDview &buffer = face.recvBuffer4;

  // MDRange3 range_face = MDRange3({0, 0, 0}, {buffer.extent(0),
  // buffer.extent(1), buffer.extent(1)});

  for (int g = 0; g < ng; g++) {
    int s = slices[g];

    threeDsubview viewSlice = getHaloSlice(view, nface, s);
    threeDsubview bufferSlice =
        Kokkos::subview(buffer, g, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

    Kokkos::deep_copy(viewSlice, bufferSlice);

    // Kokkos::parallel_for(
    //     "Copy buffer data", range_face,
    //     KOKKOS_LAMBDA(const int i, const int j, const int l) {
    //       // C
    //       viewSlice(i, j, l) = bufferSlice(i, j, l)
    //     });
  }
}
