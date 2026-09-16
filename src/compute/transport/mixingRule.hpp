// The case's viscosity mixing rule, as a piece a transport kernel calls:
// mixingRule::viscosity(X, sqrtMu) from the species' and the mole
// fractions. The jit picks the rule the config names: it forces
// transport/mixingRule/<name>.hpp in ahead of the source and defines
// PG_MIXING_RULE, so nothing here names one.
#ifndef __mixingRule_H__
#define __mixingRule_H__

#include "mixing.hpp"

#ifndef PG_MIXING_RULE
#error                                                                         \
    "a transport kernel is compiled for one mixing rule: -DPG_MIXING_RULE from the jit"
#endif

namespace mixingRule = PG_MIXING_RULE;

#endif
