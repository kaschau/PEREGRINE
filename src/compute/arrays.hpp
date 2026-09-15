// How a body reaches data: a column of the block table pinned to the cell
// the launch shape hands it (in, out, inout, and a flux kernel's sides), a
// column of a block face launch pinned to a halo cell (halo, block face and
// record columns), a
// case-wide record, the entry's dims, a per-entry value, and the window
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
KOKKOS_INLINE_FUNCTION int faceAxis(const int nface) { return (nface - 1) / 2; }
KOKKOS_INLINE_FUNCTION bool faceLow(const int nface) { return nface % 2 == 1; }

// A window is an array as its record describes it: the data pointer and
// the record, whose strides are read from memory as they are needed, so a
// thread holds two pointers per array. Everything below is plain data.
// the index the layout runs fastest, whose stride is one
KOKKOS_INLINE_FUNCTION constexpr int fastest(const int R) {
  return std::is_same_v<layout, Kokkos::LayoutLeft> ? 0 : R - 1;
}

template <class T, int R> struct window {
  T *data;
  const pgView *record;

  template <class... I> KOKKOS_INLINE_FUNCTION T &operator()(I... index) const {
    const long at[] = {static_cast<long>(index)...};
    long offset = at[fastest(R)];
    for (int d = 0; d < R; d++)
      if (d != fastest(R))
        offset += at[d] * record->stride[d];
    return data[offset];
  }
  KOKKOS_INLINE_FUNCTION int extent(const int d) const {
    return record->extent[d];
  }
  KOKKOS_INLINE_FUNCTION long stride(const int d) const {
    return record->stride[d];
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
// records, the launch shape pins it to the cell the thread is on, and the
// body indexes what remains, whose count says the rank. A column is
// declared by where the thread looks. In a cell-center launch a
// cellCenterIn/Out/InOut column is the thread's own cell, q(l), or a fixed
// neighbor of it, q(+I, l). In a cell-face launch the thread traverses the
// faces of one direction and a cell-center column looks to a side of the
// face, cellCenterL/R (LL/RR one further), qL(l), qR(l); a face is indexed
// like the cell to its right. A cellFaceIn/Out/InOut column is the face's
// own array, F(l), A(c), and has no neighbors.
template <class T, offset O> struct column {
  const pgView *records;
  const pgView *at; // pinned: the entry's record
  cell c;           // pinned: where the body is

  KOKKOS_INLINE_FUNCTION void pin(const cell &where) {
    at = records + where.entry;
    c = where;
  }
  // the element at a step from the cell: the address is made here, so the
  // compiler keeps what the body shares and nothing more
  template <class... X>
  KOKKOS_INLINE_FUNCTION T &element(const offset &o, X... rest) const {
    constexpr int R = 3 + sizeof...(X);
    assert(at->rank == R);
    const long index[] = {static_cast<long>(c.i + O.di + o.di),
                          static_cast<long>(c.j + O.dj + o.dj),
                          static_cast<long>(c.k + O.dk + o.dk),
                          static_cast<long>(rest)...};
    long offset = index[fastest(R)];
    for (int d = 0; d < R; d++)
      if (d != fastest(R))
        offset += index[d] * at->stride[d];
    return at->data[offset];
  }
  template <class... X>
    requires(std::is_integral_v<X> && ...)
  KOKKOS_INLINE_FUNCTION T &operator()(X... rest) const {
    return element(offset{0, 0, 0}, rest...);
  }
  // a fixed neighbor of the cell
  template <class... X>
  KOKKOS_INLINE_FUNCTION T &operator()(const offset &o, X... rest) const {
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
  KOKKOS_INLINE_FUNCTION T &operator()(X... rest) const {
    return this->element(offset{0, 0, 0}, rest...);
  }
  template <class... X>
  KOKKOS_INLINE_FUNCTION T &operator()(const offset &, X...) const = delete;
};
using cellFaceIn = cellFaceColumn<const double>;
using cellFaceOut = cellFaceColumn<double>;
using cellFaceInOut = cellFaceOut;

// a case-wide record (the species data), carried by value so the device
// has it, indexed as it is
struct record {
  pgView r;
  template <class... X>
  KOKKOS_INLINE_FUNCTION const double &operator()(X... index) const {
    return window<const double, sizeof...(X)>{r.data, &r}(index...);
  }
  KOKKOS_INLINE_FUNCTION int extent(const int d) const { return r.extent[d]; }
};

// one value per entry of the table (an integer column), pinned with the
// columns so the body reads it as a value
template <class T> struct perEntry {
  const T *all;
  T value;
  template <class P> KOKKOS_INLINE_FUNCTION void pin(const P &p) {
    value = all[p.entry];
  }
  KOKKOS_INLINE_FUNCTION T operator()() const { return value; }
};

// the entry's dims, pinned with the columns
struct dims {
  const pgDims *all, *at;
  KOKKOS_INLINE_FUNCTION void pin(const cell &c) { at = all + c.entry; }
  KOKKOS_INLINE_FUNCTION const pgDims *operator->() const { return at; }
};

// The columns of a block face launch, pinned to one halo cell of one of a
// block's six faces: a thread stands on one halo layer of one plane cell.
// An i face is the prototype: the halo lies to the left of the face and the
// interior to the right. A column is declared by what the thread stands
// next to. A halo column is a cell-center array of the block walked in
// layers from the face: q.L(n) is the thread's own halo cell, q.R(n),
// q.RR(n), q.RRR(n) the first three interior cells, q.at(layer, n) any
// layer counted from the face, negative into the halo. A block face column is
// a value on the block face at this plane cell, whether the array is the
// block's cell-face area vectors or the face's own values: S(c),
// qBcVals(n). A record column is a per-face constant indexed as it is, the
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
struct plane {
  int entry, g, i, j, nface;
};

// what every column of a block face launch shares: the records, the pinned face
// and plane cell, and which way the face looks
template <class T> struct atBlockFace {
  const pgView *records;
  const pgView *at_; // pinned: the face's record
  plane p;           // pinned

  KOKKOS_INLINE_FUNCTION void pin(const plane &where) {
    at_ = records + where.entry;
    p = where;
  }
  // the axis this face is normal to, and whether it is the low end of it
  KOKKOS_INLINE_FUNCTION int axis() const { return faceAxis(p.nface); }
  KOKKOS_INLINE_FUNCTION bool low() const { return faceLow(p.nface); }
  // the sign of the outward normal along the axis
  KOKKOS_INLINE_FUNCTION double outward() const { return low() ? -1.0 : 1.0; }
  KOKKOS_INLINE_FUNCTION int extent(const int d) const {
    return at_->extent[d];
  }
  // the block index along the axis of the cell `layer` from the face, the
  // first interior cell at 0, negative into the halo
  KOKKOS_INLINE_FUNCTION int along(const int layer) const {
    return low() ? ng + layer : at_->extent[axis()] - ng - 1 - layer;
  }
  // this plane cell's indices in a block array, with `along` on the axis
  template <class... X>
  KOKKOS_INLINE_FUNCTION T &inBlock(const int along, X... rest) const {
    const pgView &v = *at_;
    int b[3];
    int d = 0;
    for (int k = 0; k < 3; k++) {
      b[k] = k == axis() ? along : (d++ == 0 ? p.i : p.j);
      assert(b[k] >= 0 && b[k] < v.extent[k]);
    }
    assert(v.rank == 3 + sizeof...(X));
    return window<T, 3 + sizeof...(X)>{v.data, &v}(b[0], b[1], b[2], rest...);
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
  // the same column standing on another plane cell of this face
  KOKKOS_INLINE_FUNCTION haloColumn on(const int a, const int b) const {
    haloColumn c = *this;
    c.p.i = a, c.p.j = b;
    return c;
  }
};
using haloIn = haloColumn<const double>;
using haloOut = haloColumn<double>;
using haloInOut = haloOut; // read and written, for python's graph

// a value on the block face at this plane cell: from the block's cell-face
// array (rank 4, the plane of cell faces lying on the block face), or from the
// face's own values (rank 3)
template <class T> struct blockFaceColumn : atBlockFace<T> {
  using atBlockFace<T>::p;
  template <class... X> KOKKOS_INLINE_FUNCTION T &operator()(X... rest) const {
    const pgView &v = *this->at_;
    if (v.rank == 3 + sizeof...(X)) {
      // the block face's plane of cell faces: a cell-face array has one more
      // plane along the axis than the cell array, so its layer 0 is the
      // block face on either side
      return this->inBlock(this->along(0), rest...);
    }
    assert(v.rank == 2 + sizeof...(X));
    return window<T, 2 + sizeof...(X)>{v.data, &v}(p.i, p.j, rest...);
  }
};
using blockFaceIn = blockFaceColumn<const double>;

// a per-face array indexed as it is
template <class T> struct recordColumn : atBlockFace<T> {
  template <class... X> KOKKOS_INLINE_FUNCTION T &operator()(X... index) const {
    assert(this->at_->rank == sizeof...(X));
    return window<T, sizeof...(X)>{this->at_->data, this->at_}(index...);
  }
};
using recordIn = recordColumn<const double>;
// a face's exchange buffer, laid out (layer, a, b, components) for the
// neighbor and indexed as it is
using bufferIn = recordColumn<const double>;
using bufferOut = recordColumn<double>;

// The windows by rank, and the records as windows: the form of the three
// hand-unrolled flux schemes, until they are generalized.
using in1 = window<const double, 1>;
using in2 = window<const double, 2>;
using in3 = window<const double, 3>;
using in4 = window<const double, 4>;
using in5 = window<const double, 5>;
using out1 = window<double, 1>;
using out2 = window<double, 2>;
using out3 = window<double, 3>;
using out4 = window<double, 4>;
using out5 = window<double, 5>;

// the record as the window of the rank the kernel expects, read-only or
// writable by what the kernel said; callable where the kernels run
template <class T, int R>
KOKKOS_INLINE_FUNCTION window<T, R> asWindow(const pgView &v) {
  return {v.data, &v};
}
KOKKOS_INLINE_FUNCTION in1 as1(const pgIn &v) {
  return asWindow<const double, 1>(v);
}
KOKKOS_INLINE_FUNCTION out1 as1(const pgOut &v) {
  return asWindow<double, 1>(v);
}
KOKKOS_INLINE_FUNCTION in2 as2(const pgIn &v) {
  return asWindow<const double, 2>(v);
}
KOKKOS_INLINE_FUNCTION out2 as2(const pgOut &v) {
  return asWindow<double, 2>(v);
}
KOKKOS_INLINE_FUNCTION in3 as3(const pgIn &v) {
  return asWindow<const double, 3>(v);
}
KOKKOS_INLINE_FUNCTION out3 as3(const pgOut &v) {
  return asWindow<double, 3>(v);
}
KOKKOS_INLINE_FUNCTION in4 as4(const pgIn &v) {
  return asWindow<const double, 4>(v);
}
KOKKOS_INLINE_FUNCTION out4 as4(const pgOut &v) {
  return asWindow<double, 4>(v);
}
KOKKOS_INLINE_FUNCTION in5 as5(const pgIn &v) {
  return asWindow<const double, 5>(v);
}
KOKKOS_INLINE_FUNCTION out5 as5(const pgOut &v) {
  return asWindow<double, 5>(v);
}

// the member twins python builds a kernel's record from, laid out the same
static_assert(sizeof(cellCenterIn) == 32 && sizeof(cellFaceIn) == 32 &&
              sizeof(haloIn) == 40 && sizeof(record) == 72 &&
              sizeof(perEntry<int>) == 16 && sizeof(dims) == 16);

#endif
