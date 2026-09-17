// How a body reaches data: a column of the block table pinned to the cell
// the launch shape hands it (in, out, inout, and a flux kernel's sides), a
// column of a block face launch pinned to a halo cell (halo, block face and
// plain columns), the entry's dims, a per-entry value, and getArrayElement
// underneath them all; the case's constants, which the halo depth needs.
// Nothing here launches anything. A kernel gets this through kernel.hpp.
#ifndef __arrays_H__
#define __arrays_H__

#include "abi.hpp"
#include <cassert>
#include <type_traits>

// the case's constants, baked in by the jit
#if !defined(NS) || !defined(NE) || !defined(NG)
#error "a kernel is compiled for one case: NS, NE and NG come from the jit"
#endif
constexpr int ns = NS, ne = NE, ng = NG;

// the axis a face is normal to, and whether it is the low end of it
KOKKOS_INLINE_FUNCTION int blockFaceAxis(const int nface) {
  return (nface - 1) / 2;
}
KOKKOS_INLINE_FUNCTION bool blockFaceLow(const int nface) {
  return nface % 2 == 1;
}

// getArrayElement turns indices into an element of an array as its
// arrayInfo describes it: the data pointer and the arrayInfo, whose strides
// are read as they are needed. A column (below) does not: it pins its
// entry's arrayInfo once per team and its cell per item, and its body reads
// only data. Everything below is plain data.
// the index the layout runs fastest, whose stride is one
KOKKOS_INLINE_FUNCTION constexpr int fastest(const int R) {
  return std::is_same_v<layout, Kokkos::LayoutLeft> ? 0 : R - 1;
}

template <class T, int R> struct getArrayElement {
  T *data;
  const pgArrayInfo *arrayInfo;

  template <class... I> KOKKOS_INLINE_FUNCTION T &operator()(I... index) const {
    const long at[] = {static_cast<long>(index)...};
    long offset = at[fastest(R)];
    for (int d = 0; d < R; d++)
      if (d != fastest(R))
        offset += at[d] * arrayInfo->stride[d];
    return data[offset];
  }
  KOKKOS_INLINE_FUNCTION int extent(const int d) const {
    return arrayInfo->extent[d];
  }
  KOKKOS_INLINE_FUNCTION long stride(const int d) const {
    return arrayInfo->stride[d];
  }
};

// Where a body is: an entry of the table and a cell of it, decoded from the
// tile by the launch shape. A fixed step from a cell is an offset, and a
// column may be declared at one.
struct cell {
  int entry, i, j, k;
};
struct offset {
  int di, dj, dk;
};
KOKKOS_INLINE_FUNCTION constexpr offset operator-(const offset &o) {
  return {-o.di, -o.dj, -o.dk};
}
KOKKOS_INLINE_FUNCTION constexpr offset operator+(const offset &o) { return o; }
KOKKOS_INLINE_FUNCTION constexpr offset operator*(const int n,
                                                  const offset &o) {
  return {n * o.di, n * o.dj, n * o.dk};
}
constexpr offset I{1, 0, 0}, J{0, 1, 0}, K{0, 0, 1};

// A column of the block table as a kernel's member: python hands over the
// arrayInfos, the launch shape pins it to the cell the thread is on, and the
// body indexes what remains, whose count says the rank. A column is
// declared by where the thread looks. In a cell-center launch a
// cellCenterIn/Out/InOut column is the thread's own cell, q(l), or a fixed
// neighbor of it, q(+I, l). In a cell-face launch the thread traverses the
// faces of one direction and a cell-center column looks to a side of the
// face, cellCenterL/R (LL/RR one further), qL(l), qR(l); a face is indexed
// like the cell to its right. A cellFaceIn/Out/InOut column is the face's
// own array, F(l), A(c), and has no neighbors.
// a read of a const element, as a value, through the global address space
// on AMD: a load through a generic pointer is a flat load, which may return
// out of order, so the compiler drains every load in flight before using
// any (measured: 36 drains per trip of the viscous flux's species loop); a
// global load it counts, and overlaps
template <class T> KOKKOS_INLINE_FUNCTION decltype(auto) loadAt(T *p) {
  if constexpr (std::is_const_v<T>) {
#if defined(__HIP_DEVICE_COMPILE__)
    typedef __attribute__((address_space(1))) T *global;
    return std::remove_const_t<T>(*(global)p);
#else
    return std::remove_const_t<T>(*p);
#endif
  } else {
    return *p;
  }
}

// what a team pins once: its entry; what an item pins: its cell within it
struct entry {
  int e;
};
struct within {
  cell c;
};
template <class T, offset O> struct column {
  const pgArrayInfo *arrayInfos;
  // pinned by the team: its entry's data and strides, read from the arrayInfo
  // once; pinned by the item: its cell. Read per element instead, the
  // arrayInfo's pointer and strides were a chase in front of every load that
  // the compiler could not hoist past a kernel's stores
  T *data;
  int stride[5];
  cell c;

  KOKKOS_INLINE_FUNCTION void pin(const entry &at) {
#if defined(__HIP_DEVICE_COMPILE__)
    // arrayInfos are never written during a kernel and the entry is the team's:
    // read as constant memory, they are scalar loads the wavefront shares
    typedef __attribute__((address_space(4))) const pgArrayInfo *constant;
    const constant r = (constant)arrayInfos + at.e;
#else
    const pgArrayInfo *r = arrayInfos + at.e;
#endif
    data = r->data;
    for (int d = 0; d < 5; d++)
      stride[d] = static_cast<int>(r->stride[d]);
  }
  KOKKOS_INLINE_FUNCTION void pin(const within &at) { c = at.c; }
  // the element at a step from the cell
  template <class... X>
  KOKKOS_INLINE_FUNCTION decltype(auto) element(const offset &o,
                                                X... rest) const {
    const int r[] = {static_cast<int>(rest)..., 0, 0};
    const long i = c.i + O.di + o.di, j = c.j + O.dj + o.dj,
               k = c.k + O.dk + o.dk;
    long off;
    if constexpr (std::is_same_v<layout, Kokkos::LayoutLeft>)
      off = i + j * stride[1] + k * stride[2];
    else
      off = i * stride[0] + j * stride[1] + k * stride[2];
    if constexpr (sizeof...(X) >= 1)
      off += r[0] * stride[3];
    if constexpr (sizeof...(X) >= 2)
      off += r[1] * stride[4];
    return loadAt(data + off);
  }
  template <class... X>
    requires(std::is_integral_v<X> && ...)
  KOKKOS_INLINE_FUNCTION decltype(auto) operator()(X... rest) const {
    return element(offset{0, 0, 0}, rest...);
  }
  // element i of the entry's allocation, flat: what an elements launch reads
  KOKKOS_INLINE_FUNCTION decltype(auto) operator[](const int i) const {
    return loadAt(data + i);
  }
  // a fixed neighbor of the cell
  template <class... X>
  KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const offset &o,
                                                   X... rest) const {
    return element(o, rest...);
  }
};
using cellCenterIn = column<const double, offset{0, 0, 0}>;
using cellCenterOut = column<double, offset{0, 0, 0}>;
using cellCenterInOut = cellCenterOut; // read and written, for python's graph

// a cell-face array at the face the thread is on: the same pin and address
// as a cell-center column with no step, and no neighbors
template <class T> struct cellFaceColumn : column<T, offset{0, 0, 0}> {
  template <class... X>
    requires(std::is_integral_v<X> && ...)
  KOKKOS_INLINE_FUNCTION decltype(auto) operator()(X... rest) const {
    return this->element(offset{0, 0, 0}, rest...);
  }
  template <class... X>
  KOKKOS_INLINE_FUNCTION decltype(auto) operator()(const offset &,
                                                   X...) const = delete;
};
using cellFaceIn = cellFaceColumn<const double>;
using cellFaceOut = cellFaceColumn<double>;
using cellFaceInOut = cellFaceOut;

// one value per entry of the table (an integer column), pinned with the
// columns so the body reads it as a value
template <class T> struct perEntry {
  const T *all;
  T value;
  KOKKOS_INLINE_FUNCTION void pin(const entry &at) { value = all[at.e]; }
  KOKKOS_INLINE_FUNCTION T operator()() const { return value; }
};

// the entry's dims, pinned with the columns
struct dims {
  const pgDims *all, *at;
  KOKKOS_INLINE_FUNCTION void pin(const entry &e) { at = all + e.e; }
  KOKKOS_INLINE_FUNCTION const pgDims *operator->() const { return at; }
};

// The columns of a block face launch, pinned to one halo cell of one of a
// block's six faces: a thread stands on one halo cell, some layers out at
// one position on the block face proper (the halo edges and corners are
// never stood on).
// An i face is the prototype: the halo lies to the left of the face and the
// interior to the right. A column is declared by what the thread stands
// next to. A halo column is a cell-center array of the block walked in
// layers from the face: q.L(n) is the thread's own halo cell, q.R(n),
// q.RR(n), q.RRR(n) the first three interior cells, q.at(layer, n) any
// layer counted from the face, negative into the halo. A block face column is
// a value on the block face at this halo cell's position, whether the array is
// the block's cell-face area vectors or the face's own values: S(c),
// qBcVals(n). A plain column is an array of the face read as it is, the
// rotation matrix; a buffer column is a face's exchange buffer, the same. A bc
// declares the columns it uses, in, out or inout, and nothing else, and runs
// over every halo layer.
//
// A bc has a struct per bcHook, the point of the step it runs at: euler
// sets the halo state the inviscid fluxes see, and every wall is a slip
// wall there; preDqDxyz then makes no-slip walls correct so the velocity
// gradients on them come out right; postDqDxyz sets the halo gradients, of
// which only the first layer's are read, by the face.
// where a bc's thread stands
struct haloCell {
  int entry, g, i, j, nface;
};

// what every column of a block face launch shares: the arrayInfos, the face's
// arrayInfo pinned by the team, the halo cell pinned by the item, and which
// way the face looks. The arrayInfo stays a pointer here: a halo cell's threads
// read a few elements each, and holding the arrayInfo's extents and strides
// in the column measured 4% slower than reading them (the column copied
// per item outweighs the chase it saves)
template <class T> struct atBlockFace {
  const pgArrayInfo *arrayInfos;
  const pgArrayInfo *arrayInfo;
  haloCell p;

  KOKKOS_INLINE_FUNCTION void pin(const entry &at) {
    arrayInfo = arrayInfos + at.e;
  }
  KOKKOS_INLINE_FUNCTION void pin(const haloCell &where) { p = where; }
  // the axis this face is normal to, and whether it is the low end of it
  KOKKOS_INLINE_FUNCTION int axis() const { return blockFaceAxis(p.nface); }
  KOKKOS_INLINE_FUNCTION bool low() const { return blockFaceLow(p.nface); }
  // the sign of the outward normal along the axis
  KOKKOS_INLINE_FUNCTION double outward() const { return low() ? -1.0 : 1.0; }
  KOKKOS_INLINE_FUNCTION int extent(const int d) const {
    return arrayInfo->extent[d];
  }
  // the block index along the axis of the cell `layer` from the face, the
  // first interior cell at 0, negative into the halo
  KOKKOS_INLINE_FUNCTION int along(const int layer) const {
    return low() ? ng + layer : arrayInfo->extent[axis()] - ng - 1 - layer;
  }
  // this halo cell's position in a block array, with `along` on the axis;
  // the halo cell's position is on the block face proper, ng in from the array
  template <class... X>
  KOKKOS_INLINE_FUNCTION T &inBlock(const int along, X... rest) const {
    const pgArrayInfo &v = *arrayInfo;
    int b[3];
    int d = 0;
    for (int k = 0; k < 3; k++) {
      b[k] = k == axis() ? along : (d++ == 0 ? p.i : p.j) + ng;
      assert(b[k] >= 0 && b[k] < v.extent[k]);
    }
    assert(v.rank == 3 + sizeof...(X));
    return getArrayElement<T, 3 + sizeof...(X)>{v.data, &v}(b[0], b[1], b[2],
                                                            rest...);
  }
};

// a cell-center array of the block, walked in layers from the face
template <class T> struct haloColumn : atBlockFace<T> {
  using atBlockFace<T>::p;
  template <class... X>
  KOKKOS_INLINE_FUNCTION T &at(const int layer, X... rest) const {
    return this->inBlock(this->along(layer), rest...);
  }
  template <class... X> KOKKOS_INLINE_FUNCTION T &L(X... rest) const {
    return at(-1 - p.g, rest...);
  }
  template <class... X> KOKKOS_INLINE_FUNCTION T &R(X... rest) const {
    return at(0, rest...);
  }
  template <class... X> KOKKOS_INLINE_FUNCTION T &RR(X... rest) const {
    return at(1, rest...);
  }
  template <class... X> KOKKOS_INLINE_FUNCTION T &RRR(X... rest) const {
    return at(2, rest...);
  }
  // the same column standing at another position on this block face
  KOKKOS_INLINE_FUNCTION haloColumn on(const int a, const int b) const {
    haloColumn c = *this;
    c.p.i = a, c.p.j = b;
    return c;
  }
};
using haloIn = haloColumn<const double>;
using haloOut = haloColumn<double>;
using haloInOut = haloOut; // read and written, for python's graph

// a value on the block face at this halo cell's position: from the block's
// cell-face array (rank 4, the plane of cell faces lying on the block face), or
// from the face's own values (rank 3, shaped to the block face proper)
template <class T> struct blockFaceColumn : atBlockFace<T> {
  using atBlockFace<T>::p;
  template <class... X> KOKKOS_INLINE_FUNCTION T &operator()(X... rest) const {
    const pgArrayInfo &v = *this->arrayInfo;
    if (v.rank == 3 + sizeof...(X)) {
      // the block face's plane of cell faces: a cell-face array has one more
      // plane along the axis than the cell array, so its layer 0 is the
      // block face on either side
      return this->inBlock(this->along(0), rest...);
    }
    assert(v.rank == 2 + sizeof...(X));
    return getArrayElement<T, 2 + sizeof...(X)>{v.data, &v}(p.i, p.j, rest...);
  }
};
using blockFaceIn = blockFaceColumn<const double>;

// an array of the entry read in its own indices, no frame between: a
// block face's rotation, a trade's buffer
template <class T> struct plainColumn : atBlockFace<T> {
  template <class... X> KOKKOS_INLINE_FUNCTION T &operator()(X... index) const {
    assert(this->arrayInfo->rank == sizeof...(X));
    return getArrayElement<T, sizeof...(X)>{this->arrayInfo->data,
                                            this->arrayInfo}(index...);
  }
};
using plainIn = plainColumn<const double>;
// a face's exchange buffer, laid out (layer, a, b, components) for the
// neighbor and indexed as it is
using bufferIn = plainColumn<const double>;
using bufferOut = plainColumn<double>;

// getArrayElement by rank, and an array as one: the form of the three
// hand-unrolled flux schemes, until they are generalized.
using in1 = getArrayElement<const double, 1>;
using in2 = getArrayElement<const double, 2>;
using in3 = getArrayElement<const double, 3>;
using in4 = getArrayElement<const double, 4>;
using in5 = getArrayElement<const double, 5>;
using out1 = getArrayElement<double, 1>;
using out2 = getArrayElement<double, 2>;
using out3 = getArrayElement<double, 3>;
using out4 = getArrayElement<double, 4>;
using out5 = getArrayElement<double, 5>;

// the array as getArrayElement of the rank the kernel expects, read-only or
// writable by what the kernel said; callable where the kernels run
template <class T, int R>
KOKKOS_INLINE_FUNCTION getArrayElement<T, R>
getArrayElementOf(const pgArrayInfo &v) {
  return {v.data, &v};
}
KOKKOS_INLINE_FUNCTION in1 as1(const pgArrayIn &v) {
  return getArrayElementOf<const double, 1>(v);
}
KOKKOS_INLINE_FUNCTION out1 as1(const pgArrayOut &v) {
  return getArrayElementOf<double, 1>(v);
}
KOKKOS_INLINE_FUNCTION in2 as2(const pgArrayIn &v) {
  return getArrayElementOf<const double, 2>(v);
}
KOKKOS_INLINE_FUNCTION out2 as2(const pgArrayOut &v) {
  return getArrayElementOf<double, 2>(v);
}
KOKKOS_INLINE_FUNCTION in3 as3(const pgArrayIn &v) {
  return getArrayElementOf<const double, 3>(v);
}
KOKKOS_INLINE_FUNCTION out3 as3(const pgArrayOut &v) {
  return getArrayElementOf<double, 3>(v);
}
KOKKOS_INLINE_FUNCTION in4 as4(const pgArrayIn &v) {
  return getArrayElementOf<const double, 4>(v);
}
KOKKOS_INLINE_FUNCTION out4 as4(const pgArrayOut &v) {
  return getArrayElementOf<double, 4>(v);
}
KOKKOS_INLINE_FUNCTION in5 as5(const pgArrayIn &v) {
  return getArrayElementOf<const double, 5>(v);
}
KOKKOS_INLINE_FUNCTION out5 as5(const pgArrayOut &v) {
  return getArrayElementOf<double, 5>(v);
}

// the member twins python builds a kernel's argument from, laid out the same
static_assert(sizeof(cellCenterIn) == 56 && sizeof(cellFaceIn) == 56 &&
              sizeof(haloIn) == 40 && sizeof(perEntry<int>) == 16 &&
              sizeof(dims) == 16);

#endif
