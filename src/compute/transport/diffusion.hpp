// The case's species diffusion model, as a piece a transport kernel ends
// with: diffusion::coefficients(state, D) fills each species' mixture
// diffusion coefficient from what the kernel has in hand. The jit picks the
// model the config names, separately from the transport model: it forces
// transport/diffusion/<name>.hpp in ahead of the source and defines
// PG_DIFFUSION, so nothing here names one.
#ifndef __diffusion_H__
#define __diffusion_H__

#include "mixing.hpp"

#ifndef PG_DIFFUSION
#error                                                                         \
    "a transport kernel is compiled for one diffusion model: -DPG_DIFFUSION from the jit"
#endif

namespace diffusion = PG_DIFFUSION;

#endif
