#include "kernel.hpp"

// One bcType at one bcHook, over every face that has it, in one launch. The
// jit compiles this once per bcType and bcHook: the bcType's header is the
// forced include, and PG_BCTYPE::PG_BCHOOK is the struct in it.
#ifndef PG_BCTYPE
#error                                                                         \
    "a bc is compiled for one bcType: -DPG_BCTYPE and -DPG_BCHOOK from the jit"
#endif
#define PG_STRING_(a) #a
#define PG_STRING(a) PG_STRING_(a)

using bc = PG_BCTYPE::PG_BCHOOK;

PG_RANGE(cellCenters)
PG_ABI void pgBc(const bc &k, const pgTiling &t, const int *nface) {
  forHaloCells(PG_STRING(PG_BCTYPE) " " PG_STRING(PG_BCHOOK), t, k, nface);
}
