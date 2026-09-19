// A kernel on the cell faces of one direction, which is what every flux
// is: compiled once per direction, the jit says which, and the index
// offsets fold into the addressing. kernel.hpp with the direction on it.
#ifndef __faces_H__
#define __faces_H__

#include "kernel.hpp"

#ifndef PG_DIRECTION
#error                                                                         \
    "a flux kernel is compiled for one direction: -DPG_DIRECTION from the jit"
#endif
constexpr int iMod = PG_DIRECTION == 0, jMod = PG_DIRECTION == 1,
              kMod = PG_DIRECTION == 2;
// the face normal as a step between cells
constexpr offset N{iMod, jMod, kMod};

// a cell-center array straddling the face the thread traverses: L is the
// cell to its left, R the cell to its right, which the face is indexed
// like, LL and RR one further. With indices, that cell's element; without,
// the column as that cell sees it, for passing along
template <class T, int Rank> struct cellStrad : column<T, offset{0, 0, 0}> {
  template <offset O> KOKKOS_INLINE_FUNCTION column<T, O> at() const {
    column<T, O> s;
    s.arrayInfos = this->arrayInfos;
    s.data = this->data;
    for (int d = 0; d < 5; d++)
      s.stride[d] = this->stride[d];
    s.c = this->c;
    return s;
  }
  KOKKOS_INLINE_FUNCTION auto L() const { return at<-N>(); }
  KOKKOS_INLINE_FUNCTION auto R() const { return at<offset{0, 0, 0}>(); }
  KOKKOS_INLINE_FUNCTION auto LL() const { return at<-2 * N>(); }
  KOKKOS_INLINE_FUNCTION auto RR() const { return at<N>(); }
  template <class... X>
    requires(sizeof...(X) == Rank)
  KOKKOS_INLINE_FUNCTION decltype(auto) L(X... rest) const {
    return this->element(-N, rest...);
  }
  template <class... X>
    requires(sizeof...(X) == Rank)
  KOKKOS_INLINE_FUNCTION decltype(auto) R(X... rest) const {
    return this->element(offset{0, 0, 0}, rest...);
  }
  template <class... X>
    requires(sizeof...(X) == Rank)
  KOKKOS_INLINE_FUNCTION decltype(auto) LL(X... rest) const {
    return this->element(-2 * N, rest...);
  }
  template <class... X>
    requires(sizeof...(X) == Rank)
  KOKKOS_INLINE_FUNCTION decltype(auto) RR(X... rest) const {
    return this->element(N, rest...);
  }
};
using cellStradVecIn = cellStrad<const fpdtype, 1>;
using cellStradMatIn = cellStrad<const fpdtype, 2>;
// one value per cell, either side, read as the value
template <class T> struct cellStradScal : column<T, offset{0, 0, 0}> {
  KOKKOS_INLINE_FUNCTION T L() const { return this->element(-N); }
  KOKKOS_INLINE_FUNCTION T R() const { return this->element(offset{0, 0, 0}); }
};
using cellStradScalIn = cellStradScal<const fpdtype>;
static_assert(sizeof(cellStradVecIn) == 56 && sizeof(cellStradScalIn) == 56);

#endif
