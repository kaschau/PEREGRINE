// PEREGRINE's C ABI: what Python and the kernels agree on. Every struct
// here has a twin in peregrinepy/abi.py laid out the same, and both sides
// assert the sizes. The runtime and the kernels include this; a kernel gets
// it through kernel.hpp.
#ifndef __abi_H__
#define __abi_H__

#include <Kokkos_Core.hpp>
#include <cstddef>

// where kernels run, and the memory and layout Kokkos picks for it
using execSpace = Kokkos::DefaultExecutionSpace;
using viewSpace = execSpace::memory_space;
using layout = execSpace::array_layout;
using hostSpace = Kokkos::HostSpace;

// the module is built with hidden visibility; the ABI is what is not hidden
#define PG_ABI extern "C" __attribute__((visibility("default")))

extern "C" {
// one array as Python holds it: where, how big, and its strides in
// elements, which Python works out from the device layout
struct pgView {
  double *data;
  int rank;
  int extent[5];
  long stride[5];
};

// the cells of one entry a launch does: a start and an extent per axis, the
// components, and the item count
struct pgCells {
  int start[3], extent[4], n;
};

// a block's shape: cells per direction
struct pgDims {
  int ni, nj, nk;
};

// One launch over every entry of a table. Each entry has some items -- its
// cells, or the elements of its planes -- and a team does a tile of one
// entry's, so a rank of thousands of small entries and one of a few huge
// ones both fill the device. Python tiles the range a kernel declares and
// hands the tiling over with the columns: the entry of each tile, the first
// tile and the items of each entry, each entry's cells, and the items a
// tile is, the case's knob for this kind of item.
struct pgTiling {
  const int *entry, *first, *items;
  const pgCells *cells;
  int count, tiles, tile;
};
}

static_assert(sizeof(pgView) == 72 && offsetof(pgView, rank) == 8 &&
              offsetof(pgView, extent) == 12 && offsetof(pgView, stride) == 32);
static_assert(sizeof(pgCells) == 32 && offsetof(pgCells, extent) == 12 &&
              offsetof(pgCells, n) == 28);
static_assert(sizeof(pgDims) == 12);
static_assert(sizeof(pgTiling) == 48 && offsetof(pgTiling, cells) == 24 &&
              offsetof(pgTiling, count) == 32 &&
              offsetof(pgTiling, tile) == 40);

// a kernel names each record it takes by what it does with it: the same
// bytes, but an in unpacks to a view that cannot be written, and the step's
// graph orders kernels by what they read and write
struct pgIn : pgView {};
struct pgOut : pgView {};

#endif
