// PEREGRINE's C ABI: what Python and the kernels agree on. Every struct
// here has a twin in peregrinepy/backend/abi.py laid out the same, and both
// sides assert the sizes. The runtime and the kernels include this; a kernel
// gets it through kernel.hpp.
#ifndef __abi_H__
#define __abi_H__

#include <Kokkos_Core.hpp>
#include <Kokkos_Graph.hpp>
#include <cstddef>

// where kernels run, and the memory and layout Kokkos picks for it
using execSpace = Kokkos::DefaultExecutionSpace;
using viewSpace = execSpace::memory_space;
using layout = execSpace::array_layout;
using hostSpace = Kokkos::HostSpace;

// the module is built with hidden visibility; the ABI is what is not hidden
#define PG_ABI extern "C" __attribute__((visibility("default")))

// the precision of every array and kernel value: the case's, baked in by
// the jit; the runtime holds the arrays as bytes and never reads one
#ifndef PG_FPDTYPE
#define PG_FPDTYPE double
#endif
using fpdtype = PG_FPDTYPE;

// A device graph under capture: the runtime holds the graph and the node
// its next launch follows; a launch shape, finding one open, adds its
// kernel as the next node instead of launching it. The node is held type
// erased, so any kernel library can extend it.
using graphNode =
    Kokkos::Experimental::GraphNodeRef<execSpace,
                                       Kokkos::Experimental::TypeErasedTag,
                                       Kokkos::Experimental::TypeErasedTag>;
PG_ABI graphNode *pgGraphTail();
// whether a capture is open, which a reduction has no place in
PG_ABI bool pgGraphCapturing();
// independent launches under capture: siblings off one node, joined after
PG_ABI void pgGraphFork();
PG_ABI void pgGraphSibling();
PG_ABI void pgGraphJoin();

extern "C" {
// all a kernel needs to know about one array, as Python holds it: where,
// how big, and its strides in elements, which Python works out from the
// device layout
struct pgArrayInfo {
  fpdtype *data;
  int rank;
  int extent[5];
  long stride[5];
};

// one range of cells a launch does: a start and an extent per axis, the
// components, the item count, and the entry the range is of
struct pgRange {
  int start[3], extent[4], n, entry;
};

// a block's shape: cells per direction
struct pgDims {
  int ni, nj, nk;
};

// One launch over a table. A range is ranges of items -- an entry's interior
// cells, a halo slab behind one of its block faces, the elements of a
// buffer's planes -- and a team does a tile of one range's, so a rank of
// thousands of small entries and one of a few huge ones both fill the
// device, and the halo edges and corners are never generated. Python tiles
// the range a launch names and hands the tiling over with the columns: the
// range of each tile, the first tile and the items of each range, each range
// with its entry, and the items a tile is, the case's knob for this
// kind of item.
struct pgTiling {
  const int *range, *first, *items;
  const pgRange *ranges;
  int count, tiles, tile;
};
}

static_assert(sizeof(pgArrayInfo) == 72 && offsetof(pgArrayInfo, rank) == 8 &&
              offsetof(pgArrayInfo, extent) == 12 &&
              offsetof(pgArrayInfo, stride) == 32);
static_assert(sizeof(pgRange) == 36 && offsetof(pgRange, extent) == 12 &&
              offsetof(pgRange, n) == 28 && offsetof(pgRange, entry) == 32);
static_assert(sizeof(pgDims) == 12);
static_assert(sizeof(pgTiling) == 48 && offsetof(pgTiling, ranges) == 24 &&
              offsetof(pgTiling, count) == 32 &&
              offsetof(pgTiling, tile) == 40);

// a kernel names each array it takes by what it does with it: the same
// bytes, but an in unpacks to a view that cannot be written, and the step's
// graph orders kernels by what they read and write
struct pgArrayIn : pgArrayInfo {};
struct pgArrayOut : pgArrayInfo {};

#endif
