// What a formula writes its flux through: the face's column as it is, or a
// share of a combination -- set to its weight of the value, or that added.
// Either takes a column or another share, so combinations nest: a fourth
// order formula's shares inside the blend's.
#ifndef __fluxOut_H__
#define __fluxOut_H__

#include "faces.hpp"

template <class Out> struct weighted {
  const Out &F;
  fpdtype w;
  struct slot {
    decltype(std::declval<const Out &>()(0)) f;
    fpdtype w;
    KOKKOS_INLINE_FUNCTION void operator=(fpdtype v) const { f = w * v; }
    KOKKOS_INLINE_FUNCTION void operator+=(fpdtype v) const { f += w * v; }
  };
  KOKKOS_INLINE_FUNCTION slot operator()(int l) const { return {F(l), w}; }
};
template <class Out> struct added {
  const Out &F;
  fpdtype w;
  struct slot {
    decltype(std::declval<const Out &>()(0)) f;
    fpdtype w;
    KOKKOS_INLINE_FUNCTION void operator=(fpdtype v) const { f += w * v; }
    KOKKOS_INLINE_FUNCTION void operator+=(fpdtype v) const { f += w * v; }
  };
  KOKKOS_INLINE_FUNCTION slot operator()(int l) const { return {F(l), w}; }
};

#endif
