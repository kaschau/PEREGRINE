// The species data of the case, baked by the jit: a value per species, or a
// polynomial in ln T per species (ascending coefficients, every fit its own
// degree, nothing padded), reached through one accessor each. The jit
// forces the data in ahead of any source that includes this; each array is
// a static constexpr array inside a plain inline function, the one form
// both device compilers place well (a namespace-scope constexpr array is a
// host object to nvcc, and a non-static local one is rebuilt on the stack
// per call). Included by the kernels that read species data, and only them.
#ifndef __species_H__
#define __species_H__

#include "arrays.hpp"

#ifndef PG_SPECIES_DATA
#error                                                                         \
    "a species kernel is compiled with the case's species data, forced in by the jit"
#endif

constexpr fpdtype Ru = PG_RU;

// one value per species
#define PG_SPECIES_SCALAR(name, DATA)                                          \
  KOKKOS_INLINE_FUNCTION fpdtype name(const int n) {                           \
    static constexpr fpdtype t[ns] = DATA;                                     \
    return t[n];                                                               \
  }

// one value per ordered pair of species, row-major
#define PG_SPECIES_PAIR(name, DATA)                                            \
  KOKKOS_INLINE_FUNCTION fpdtype name(const int n, const int m) {              \
    static constexpr fpdtype t[ns * ns] = DATA;                                \
    return t[n * ns + m];                                                      \
  }

// one polynomial per row, every row the array's number of terms (the high
// powers a row lacks are zero, which costs Horner nothing): coefficient m
// of row r, rows contiguous, so a walk is one trip count and one load a term
#define PG_SPECIES_POLYNOMIAL(name, TERMS, DATA)                               \
  constexpr int name##Terms = TERMS;                                           \
  KOKKOS_INLINE_FUNCTION fpdtype name(const int r, const int m) {              \
    static constexpr fpdtype c[] = DATA;                                       \
    return c[r * TERMS + m];                                                   \
  }

PG_SPECIES_SCALAR(MW, PG_MW)
PG_SPECIES_SCALAR(MWinv, PG_MWINV)
PG_SPECIES_SCALAR(MWqInv, PG_MWQ_INV)
PG_SPECIES_SCALAR(sqrtMW, PG_SQRT_MW)
// Wilke's pair constant 1 / sqrt(8 (1 + MW_n / MW_m))
PG_SPECIES_PAIR(wilkePair, PG_WILKE_PAIR)
PG_SPECIES_SCALAR(hRef, PG_H_REF)
PG_SPECIES_SCALAR(cp0, PG_CP0)
PG_SPECIES_SCALAR(mu0, PG_MU0)
PG_SPECIES_SCALAR(kappa0, PG_KAPPA0)
// the Lewis number, the name the diffusion model namespace does not take
PG_SPECIES_SCALAR(lewisNumber, PG_LEWIS)
PG_SPECIES_SCALAR(Tcrit, PG_TCRIT)
PG_SPECIES_SCALAR(pcrit, PG_PCRIT)
PG_SPECIES_SCALAR(Vcrit, PG_VCRIT)
PG_SPECIES_SCALAR(acentric, PG_ACENTRIC)
PG_SPECIES_SCALAR(redDipole, PG_RED_DIPOLE)

PG_SPECIES_POLYNOMIAL(cpPoly, PG_CP_POLY_TERMS, PG_CP_POLY)
PG_SPECIES_POLYNOMIAL(hPoly, PG_H_POLY_TERMS, PG_H_POLY)
PG_SPECIES_POLYNOMIAL(sPoly, PG_S_POLY_TERMS, PG_S_POLY)
PG_SPECIES_POLYNOMIAL(muPoly, PG_MU_POLY_TERMS, PG_MU_POLY)
PG_SPECIES_POLYNOMIAL(kappaPoly, PG_KAPPA_POLY_TERMS, PG_KAPPA_POLY)
PG_SPECIES_POLYNOMIAL(chungA, PG_CHUNG_A_TERMS, PG_CHUNG_A)
PG_SPECIES_POLYNOMIAL(chungB, PG_CHUNG_B_TERMS, PG_CHUNG_B)

// the binary diffusion fits are one per unordered pair, the rows the pairs
// i <= j in numpy's triu_indices order: row i's pairs start after the
// i rows before it, which held ns, ns - 1, ... pairs
PG_SPECIES_POLYNOMIAL(dijPair, PG_DIJ_TERMS, PG_DIJ)
KOKKOS_INLINE_FUNCTION int pairIndex(const int n, const int n2) {
  const int i = n < n2 ? n : n2, j = n < n2 ? n2 : n;
  return i * ns - i * (i - 1) / 2 + (j - i);
}
constexpr int dijTerms = dijPairTerms;
KOKKOS_INLINE_FUNCTION fpdtype dij(const int n, const int n2, const int m) {
  return dijPair(pairIndex(n, n2), m);
}

#endif
