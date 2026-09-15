// PEREGRINE's C ABI: what Python and the kernels agree on. Every struct
// here has a twin in python (abi.View, abi.pgCells, table.Tiling, and the
// kernel record python builds from a kernel's members), laid out the same.
// The runtime and the kernels include this; a kernel gets it through
// kernel.hpp.
#ifndef __abi_H__
#define __abi_H__

#include <Kokkos_Core.hpp>

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
}

// One launch over every entry of a table. Each entry has some items -- its
// cells, or the elements of its planes -- and a team does `tileSize` of one
// entry's, so a rank of thousands of small entries and one of a few huge
// ones both fill the device. Python tiles the range a kernel declares and
// hands the tiling over with the columns.
constexpr int tileSize = 128; // as Tiling.tileSize in python
using teams = Kokkos::TeamPolicy<execSpace>;
using team = teams::member_type;

// the tiling python made: the entry of each tile, the first tile and the
// items of each entry, and each entry's cells; laid out as python's Tiling
struct pgTiling {
  const int *entry, *first, *items;
  const pgCells *cells;
  int count, tiles;

  struct tile {
    int e, begin, end;
  };
  // the entry and item range of this team's tile
  KOKKOS_INLINE_FUNCTION tile of(const team &t) const {
    const int rank = t.league_rank();
    const int lo = entry[rank];
    const int begin = (rank - first[lo]) * tileSize;
    const int n = items[lo];
    return {lo, begin, begin + tileSize < n ? begin + tileSize : n};
  }
  teams policy() const { return teams(tiles, Kokkos::AUTO); }
};

#endif
