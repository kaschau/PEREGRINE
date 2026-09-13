// PEREGRINE's C ABI. Python holds every array as a pointer and its extents;
// a kernel is a C function that takes the records of the arrays it reads and
// writes and unpacks each into an unmanaged Kokkos view.
#ifndef __abi_H__
#define __abi_H__

#include "kokkosTypes.hpp"
#include <cassert>

// the module is built with hidden visibility; the ABI is what is not hidden
#define PG_ABI extern "C" __attribute__((visibility("default")))

extern "C" {
// one array as Python holds it
struct pgView {
  double *data;
  int rank;
  int extent[5];
};
}

// a kernel names each record it takes by what it does with it: the same
// bytes, but an in unpacks to a view that cannot be written, and the step's
// graph orders kernels by what they read and write
struct pgIn : pgView {};
struct pgOut : pgView {};

extern "C" {
// a block's shape: cells per direction
struct pgDims {
  int ni, nj, nk;
};

// the cells a kernel does, as Python states them
struct pgRange {
  int i0, i1, j0, j1, k0, k1;
};
}

// the record as a view of the rank the kernel expects, read-only or writable
// by what the kernel said
inline in1 as1(const pgIn &v) {
  assert(v.rank == 1);
  return in1(v.data, v.extent[0]);
}
inline out1 as1(const pgOut &v) {
  assert(v.rank == 1);
  return out1(v.data, v.extent[0]);
}
inline in2 as2(const pgIn &v) {
  assert(v.rank == 2);
  return in2(v.data, v.extent[0], v.extent[1]);
}
inline out2 as2(const pgOut &v) {
  assert(v.rank == 2);
  return out2(v.data, v.extent[0], v.extent[1]);
}
inline in3 as3(const pgIn &v) {
  assert(v.rank == 3);
  return in3(v.data, v.extent[0], v.extent[1], v.extent[2]);
}
inline out3 as3(const pgOut &v) {
  assert(v.rank == 3);
  return out3(v.data, v.extent[0], v.extent[1], v.extent[2]);
}
inline in4 as4(const pgIn &v) {
  assert(v.rank == 4);
  return in4(v.data, v.extent[0], v.extent[1], v.extent[2], v.extent[3]);
}
inline out4 as4(const pgOut &v) {
  assert(v.rank == 4);
  return out4(v.data, v.extent[0], v.extent[1], v.extent[2], v.extent[3]);
}
inline in5 as5(const pgIn &v) {
  assert(v.rank == 5);
  return in5(v.data, v.extent[0], v.extent[1], v.extent[2], v.extent[3],
             v.extent[4]);
}
inline out5 as5(const pgOut &v) {
  assert(v.rank == 5);
  return out5(v.data, v.extent[0], v.extent[1], v.extent[2], v.extent[3],
              v.extent[4]);
}

#endif
