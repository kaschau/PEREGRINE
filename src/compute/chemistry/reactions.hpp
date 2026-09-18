// The reactions of the case, baked by the jit like the species data: a
// value per reaction, or a row of (species, value) terms per reaction
// padded with (0, 0), reached through one accessor each. Two forms: a
// constexpr accessor for what a reaction's body is composed on -- its
// type, its reversibility, which of a rate's terms exist -- and the
// static-table form of species.hpp for what it reads.
#ifndef __reactions_H__
#define __reactions_H__

#include "species.hpp"

#ifndef PG_REACTIONS_DATA
#error                                                                         \
    "a chemistry kernel is compiled with the case's reactions, forced in by the jit"
#endif

namespace reactions {

constexpr int nr = PG_NR;
enum kind { elementary = 0, threeBody = 1, lindemann = 2, troe = 3 };
constexpr fpdtype logPrefRu = PG_LOG_PREF_RU;
constexpr fpdtype logCFloor = PG_LOG_CFLOOR;
constexpr fpdtype maxOmegaGain = PG_MAX_OMEGA_GAIN;

// a value per reaction the composition selects on: a constant expression
#define PG_REACTION_CONSTANT(name, type, DATA)                                 \
  KOKKOS_INLINE_FUNCTION constexpr type name(const int r) {                    \
    constexpr type v[nr] = DATA;                                               \
    return v[r];                                                               \
  }
// a value per reaction a rate reads
#define PG_REACTION_SCALAR(name, DATA)                                         \
  KOKKOS_INLINE_FUNCTION fpdtype name(const int r) {                           \
    static constexpr fpdtype t[nr] = DATA;                                     \
    return t[r];                                                               \
  }
// a row of terms per reaction, every row the table's number of terms
#define PG_REACTION_TERMS(name, type, TERMS, DATA)                             \
  constexpr int name##Terms = TERMS;                                           \
  KOKKOS_INLINE_FUNCTION type name(const int r, const int k) {                 \
    static constexpr type t[] = DATA;                                          \
    return t[r * TERMS + k];                                                   \
  }

PG_REACTION_CONSTANT(type, int, PG_TYPE)
PG_REACTION_CONSTANT(reversible, int, PG_REVERSIBLE)
// ln k_f = logA + b ln T - EaR / T; and k0 / kinf of a falloff likewise
PG_REACTION_CONSTANT(logA, fpdtype, PG_LOG_A)
PG_REACTION_CONSTANT(b, fpdtype, PG_B)
PG_REACTION_CONSTANT(EaR, fpdtype, PG_EA_R)
PG_REACTION_CONSTANT(logPrA, fpdtype, PG_LOG_PR_A)
PG_REACTION_CONSTANT(prB, fpdtype, PG_PR_B)
PG_REACTION_CONSTANT(prEaR, fpdtype, PG_PR_EA_R)
PG_REACTION_CONSTANT(fcentTerms, int, PG_FCENT_TERMS)
PG_REACTION_SCALAR(defaultEfficiency, PG_DEFAULT_EFFICIENCY)
PG_REACTION_SCALAR(nuTotal, PG_NU_TOTAL)
// Troe's centering, each term +-exp(c0 + cT T + cTinv / T)
PG_REACTION_TERMS(fcentSign, fpdtype, PG_FCENT_SIGN_TERMS, PG_FCENT_SIGN)
PG_REACTION_TERMS(fcentC0, fpdtype, PG_FCENT_C0_TERMS, PG_FCENT_C0)
PG_REACTION_TERMS(fcentCT, fpdtype, PG_FCENT_CT_TERMS, PG_FCENT_CT)
PG_REACTION_TERMS(fcentCTinv, fpdtype, PG_FCENT_CTINV_TERMS, PG_FCENT_CTINV)
// the forward exponents, the net stoichiometry, and the third-body
// deviations from the default efficiency, each (species, value)
PG_REACTION_TERMS(fwdSpecies, int, PG_FWD_SPECIES_TERMS, PG_FWD_SPECIES)
PG_REACTION_TERMS(fwdExponent, fpdtype, PG_FWD_EXPONENT_TERMS, PG_FWD_EXPONENT)
PG_REACTION_TERMS(netSpecies, int, PG_NET_SPECIES_TERMS, PG_NET_SPECIES)
PG_REACTION_TERMS(netNu, fpdtype, PG_NET_NU_TERMS, PG_NET_NU)
PG_REACTION_TERMS(netNuMW, fpdtype, PG_NET_NU_MW_TERMS, PG_NET_NU_MW)
PG_REACTION_TERMS(effSpecies, int, PG_EFF_SPECIES_TERMS, PG_EFF_SPECIES)
PG_REACTION_TERMS(effDeviation, fpdtype, PG_EFF_DEVIATION_TERMS,
                  PG_EFF_DEVIATION)

} // namespace reactions

#endif
