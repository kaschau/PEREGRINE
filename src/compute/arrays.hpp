// How a body reaches data: a column of the block table pinned to the cell
// the launch shape hands it (in, out, inout, and a flux kernel's sides), a
// column of the face table pinned to a halo cell (faceIn, faceOut), a
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

// A column of the table as a kernel's member: python hands over the records,
// the launch shape pins it to the cell, and the body indexes what remains,
// whose count says the rank. A read-only column is an in, a written one an
// out (inout when it is both, for python's graph); a flux kernel's are
// declared a side away from the face.
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
using in = column<const double, offset{0, 0, 0}>;
using out = column<double, offset{0, 0, 0}>;
using inout = out;

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

// A column of the face table as a hook's member, pinned to one halo cell of
// one face. An i face is the prototype: the halo lies to the left of the
// face and the interior to the right. A thread stands on one halo layer of
// one plane cell: q.L(n) is its own halo cell, q.R(n), q.RR(n), q.RRR(n) the
// first three interior cells, the words a wide flux stencil uses; q.at(layer,
// n) reaches any layer, counted from the face, negative into the halo. A
// face array's own plane is its R; a face's values are read here(n), a
// per-face record as it is. A hook declares the columns it uses, faceIn,
// faceOut or faceInOut, and nothing else. Every hook runs over every halo
// layer.
//
// The hooks: euler sets the halo state the inviscid fluxes see, and every
// wall is a slip wall there; preDqDxyz then makes no-slip walls correct so
// the velocity gradients on them come out right; postDqDxyz sets the halo
// gradients, of which only the first layer's are read, by the face.
// where a hook's thread stands
struct plane {
  int entry, g, i, j, nface;
};

template <class T> struct faceColumn {
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

  // an element at a layer from the face, in this plane cell
  template <class... X>
  KOKKOS_INLINE_FUNCTION T &at(const int layer, X... rest) const {
    const pgView &v = *at_;
    const int a = axis();
    const int along = low() ? ng + layer : v.extent[a] - ng - 1 - layer;
    int b[3];
    int d = 0;
    for (int k = 0; k < 3; k++) {
      b[k] = k == a ? along : (d++ == 0 ? p.i : p.j);
      assert(b[k] >= 0 && b[k] < v.extent[k]);
    }
    assert(v.rank == 3 + sizeof...(X));
    return window<T, 3 + sizeof...(X)>{v.data, &v}(b[0], b[1], b[2], rest...);
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
  KOKKOS_INLINE_FUNCTION faceColumn on(const int a, const int b) const {
    faceColumn c = *this;
    c.p.i = a, c.p.j = b;
    return c;
  }
  KOKKOS_INLINE_FUNCTION int extent(const int d) const {
    return at_->extent[d];
  }
  // a face's own values, at this plane cell
  template <class... X> KOKKOS_INLINE_FUNCTION T &here(X... rest) const {
    assert(at_->rank == 2 + sizeof...(X));
    return window<T, 2 + sizeof...(X)>{at_->data, at_}(p.i, p.j, rest...);
  }
  // a per-face record, indexed as it is
  template <class... X> KOKKOS_INLINE_FUNCTION T &operator()(X... index) const {
    assert(at_->rank == sizeof...(X));
    return window<T, sizeof...(X)>{at_->data, at_}(index...);
  }
};
using faceIn = faceColumn<const double>;
using faceOut = faceColumn<double>;
using faceInOut = faceOut; // read and written, for python's graph

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

#endif
