// PEREGRINE's C ABI. Python holds every array as a pointer and its extents;
// a kernel is a C function that takes the records of the arrays it reads and
// writes and unpacks each into an unmanaged Kokkos view.
#ifndef __abi_H__
#define __abi_H__

#include "kokkosTypes.hpp"

// the module is built with hidden visibility; the ABI is what is not hidden
#define PG_ABI extern "C" __attribute__((visibility("default")))

extern "C" {
// one array as Python holds it
struct pgView {
  double *data;
  int rank;
  int extent[5];
};

// a block's shape: cells per direction
struct pgDims {
  int ni, nj, nk;
};

// the cells a kernel does, as Python states them
struct pgRange {
  int i0, i1, j0, j1, k0, k1;
};
}

inline unmanaged<double *> as1(const pgView &v) {
  return unmanaged<double *>(v.data, v.extent[0]);
}
inline unmanaged<double **> as2(const pgView &v) {
  return unmanaged<double **>(v.data, v.extent[0], v.extent[1]);
}
inline unmanaged<double ***> as3(const pgView &v) {
  return unmanaged<double ***>(v.data, v.extent[0], v.extent[1], v.extent[2]);
}
inline unmanaged<double ****> as4(const pgView &v) {
  return unmanaged<double ****>(v.data, v.extent[0], v.extent[1], v.extent[2],
                                v.extent[3]);
}
inline unmanaged<double *****> as5(const pgView &v) {
  return unmanaged<double *****>(v.data, v.extent[0], v.extent[1], v.extent[2],
                                 v.extent[3], v.extent[4]);
}

#endif
